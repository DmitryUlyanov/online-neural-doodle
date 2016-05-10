require 'nn'
require 'cunn'
require 'cutorch'
require 'image'
require 'optim'
require 'hdf5'

display = require('display')

require 'src/utils'
require 'src/descriptor_net'

local cmd = torch.CmdLine()

cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'Layers to attach texture loss.')

cmd:option('-masks_hdf5', '', 
           'Path to .hdf5 file with masks. It can be obtained with get_mask_hdf5.py.')

cmd:option('-learning_rate', 1e-1)
cmd:option('-num_iterations', 5000)

cmd:option('-batch_size', 4)

cmd:option('-num_mask_noise_times', 1, 'Number of channels of the input tensor.')
cmd:option('-num_noise_channels', 0, 'Number of channels of the input tensor.')

cmd:option('-tmp_path', 'data/out/', 'Directory to store intermediate results.')
cmd:option('-model_name', '', 'Path to generator model description file.')

cmd:option('-normalize_gradients', 'false', 'L1 gradient normalization inside descriptor net. ')
cmd:option('-vgg_no_pad', 'false')

cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt', 'Pretrained')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-half', 'true', 'If true adds another VGG instance which computes loss and gradient at 0.5 scale.')

cmd:option('-pooling', 'avg', 'Pooling method to use')


params = cmd:parse(arg)

params.half = params.half ~= 'false'
params.vgg_no_pad = params.vgg_no_pad ~= 'false'
params.circular_padding = params.circular_padding ~= 'false'

local train_hdf5 = hdf5.open(params.masks_hdf5)
local style_img = train_hdf5:read('style_img'):all()
local style_mask = train_hdf5:read('style_mask'):all()

n_colors = style_mask:size(1)

local width = style_img:size(2)
local height = style_img:size(3)


if params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.fastest = true
  cudnn.benchmark = true
  backend = cudnn
else
  backend = nn
end


conv = convc

-- Load VGG
if params.backend == 'clnn' then params.backend = 'nn' end
cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):cuda()
for i = 1,9 do
  cnn:remove()
end


net_input_depth = params.num_mask_noise_times*n_colors + n_colors + params.num_noise_channels

-- Define model
local net = require('models/' .. params.model_name):cuda()

-- Setup descriptor net
descriptor_net, style_losses, mask_net, masks_modules = create_loss_net(params)

if params.half then
  descriptor_net_half, style_losses_half, mask_net_half, masks_modules_half = create_loss_net(params, true)
end

cnn = nil
collectgarbage()

----------------------------------------------------------
-- feval
----------------------------------------------------------


iteration = 0

-- dummy storage, this will not be changed during training
masks_batch = torch.Tensor(params.batch_size, n_colors, width, height)
means_batch = torch.Tensor(params.batch_size, n_colors):uniform()

inputs_batch = torch.Tensor(params.batch_size, n_colors, width, height):cuda()

n_train = train_hdf5:read('n_train'):all()[1]

cur_index_train = 0
function get_masks_train()
  -- Ignore last for simplicity
  if cur_index_train > n_train - params.batch_size then
    cur_index_train = 0 
  end

  for i = 0, params.batch_size-1 do
    masks_batch[i+1]:narrow(1,1,n_colors):copy(train_hdf5:read('train_mask_' .. (cur_index_train + i)):all())
  end

  cur_index_train = cur_index_train + params.batch_size

  return masks_batch:cuda()
end


local parameters, gradParameters = net:getParameters()
loss_history = {}
function feval(x)
  iteration = iteration + 1
  
  if x ~= parameters then
      parameters:copy(x)
  end
  gradParameters:zero()
  
  local masks_batch_ = get_masks_train()

  inputs_batch:copy(masks_batch_/10)

  -- forward
  local out = net:forward(inputs_batch)
  
  --------------------------------
  -- Full
  --------------------------------
  mask_net:forward(masks_batch_)
  for i =1, #masks_modules do 
    style_losses[i].target_masks = masks_modules[i].output
  end

  descriptor_net:forward(out)
  local gradFull = descriptor_net:backward(out, nil)
  
  ------------------------------------
  -- Half 
  ------------------------------------
  if params.half then 
    mask_net_half:forward(masks_batch_)
    for i =1, #masks_modules_half do 
      style_losses_half[i].target_masks = masks_modules_half[i].output
    end

    descriptor_net_half:forward(out)
    local gradHalf = descriptor_net_half:backward(out, nil)

    net:backward(inputs_batch, gradFull + gradHalf)
    else
    net:backward(inputs_batch, gradFull)  
  end
   
  -- collect loss
  local loss = 0
  for _, mod in ipairs(style_losses) do
    loss = loss + mod.loss
  end
  if params.half then
    for _, mod in ipairs(style_losses_half) do
      loss = loss + mod.loss
    end
  end
  
  table.insert(loss_history, {iteration,loss/params.batch_size})
  print(iteration, loss/params.batch_size)

  return loss, gradParameters
end
----------------------------------------------------------
-- Optimize
----------------------------------------------------------
print('        Optimize        ')

optim_method = optim.adam
state = {
   learningRate = params.learning_rate,
}


for it = 1, params.num_iterations do
  -- Optimization step
  optim_method(feval, parameters, state)

  -- Visualize
  if it%10 == 0 then
    collectgarbage()

    local output = net.output:clone():double()

    local imgs  = {}
    for i = 1, output:size(1) do
      local img = deprocess(output[i])
      table.insert(imgs, torch.clamp(img,0,1))
      image.save(params.tmp_path .. 'train' .. i .. '_' .. it .. '.png',img)
    end

    display.image(imgs, {win=1,  title = 'Preview'})
    display.plot(loss_history, {win=2, labels={'iteration', 'Loss'}, title='Loss'})
  end
  
  if it%300 == 0 then 
    state.learningRate = state.learningRate*0.8 
  end

  if it%200 == 0 then 
    torch.save(params.tmp_path .. 'model' .. it .. '.t7', net:clearState())
  end
end

-- Clean net and dump it
torch.save(params.tmp_path .. 'model.t7', net:clearState())