require 'src/style_loss'

require 'loadcaffe'

function nop()
  -- nop.  not needed by our net
end

function create_loss_net(params, half)
  
  local half = half or false
-- Load style
  local f_data = hdf5.open(params.masks_hdf5)
  local style_img = f_data:read('style_img'):all()
  
  style_img = preprocess(style_img):add_dummy():cuda()
  


  -- Of size M x W x H 
  local style_mask = f_data:read('style_mask'):all():cuda()
  local n_colors = style_mask:size(1)

 
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  local style_losses, masks_modules = {}, {}
  local next_style_idx = 1
  local net = nn.Sequential()



  local mask_net_avg = nn.Sequential():add(nn.Identity())

  if half then 
    net:add(nn.SpatialAveragePooling(2,2,2,2):cuda())
    mask_net_avg:add(nn.SpatialAveragePooling(2,2,2,2):cuda())
  end
  local cnn1 = cnn:clone()
  for i = 1, #cnn1 do

    if next_style_idx <= #style_layers then
      local layer = cnn1:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      local is_conv =  (layer_type == 'nn.SpatialConvolution' or layer_type == 'cudnn.SpatialConvolution')
     
      -------------------------------------------------
      -- Pooling
      -------------------------------------------------
      if is_pooling then
        
        if params.pooling == 'avg' then
          assert(layer.padW == 0 and layer.padH == 0)
          local kW, kH = layer.kW, layer.kH
          local dW, dH = layer.dW, layer.dH
          local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):cuda()
          avg_pool_layer:cuda()
          
          local msg = 'Replacing max pooling at layer %d with average pooling'
          print(string.format(msg, i))
          
          layer = avg_pool_layer
        end



        mask_net_avg:add(nn.SpatialAveragePooling(2,2,2,2):cuda())
      -------------------------------------------------
      -- Convolution
      -------------------------------------------------
      elseif is_conv then

        -- Turn off padding
        if params.vgg_no_pad and (layer_type == 'nn.SpatialConvolution' or layer_type == 'cudnn.SpatialConvolution') then
          layer.padW = 0
          layer.padH = 0
          
          mask_net_avg:add(nn.SpatialZeroPadding(-1,-1,-1,-1):cuda())
        else
          mask_net_avg:add(nn.SpatialAveragePooling(3,3,1,1,1,1):cuda())
        end
      end

      net:add(layer)
      
      -------------------------------------------------
      -- Style loss
      -------------------------------------------------
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():cuda()

        -- 1 x C x W x H
        local target_features = net:forward(style_img):clone()
        -- M x W x H
        local layer_masks = mask_net_avg:forward(style_mask)

        -- Compute target gram mats
        local target_grams = {}
        for k = 1, n_colors do

          -- 1 x 1 x W x H
          local mask = layer_masks:narrow(1,k,1):add_dummy()
          -- 1 x C x W x H
          -- print(target_features:size(), mask:size(), mask_net)
          local exp_mask = mask:expandAs(target_features)
          -- 1 x C x W x H
          local masked = torch.cmul(target_features, exp_mask)
          -- 1 x C x C
          local target = gram:forward(masked):clone()

          target:div(target_features[1]:nElement() * (mask:mean()+1e-6))
          
          target_grams[k] = target
        end

        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target_grams, norm):cuda()
      
        table.insert(masks_modules, mask_net_avg.modules[#mask_net_avg.modules])
        table.insert(style_losses, loss_module)
        
        net:add(loss_module)
                
        next_style_idx = next_style_idx + 1
      end
    end
  end
  init = false
  -- We don't need the base CNN anymore, so clean it up to save memory.
    
  net:add(nn.DummyGradOutput())

  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()



  -- print(net)
  

  -- -- We don't need the base CNN anymore, so clean it up to save memory.
  -- cnn = nil
  -- for i=1, #net.modules do
  --   local module = net.modules[i]
  --   if torch.type(module) == 'nn.SpatialConvolutionMM' or torch.type(module) == 'nn.SpatialConvolution' or torch.type(module) == 'cudnn.SpatialConvolution' then
  --       module.gradWeight = nil
  --       module.gradBias = nil
  --   end
  -- end
  -- collectgarbage()
  

  -- -- Initialize with previous or with noise
  -- if img then 
  --   img = image.scale(img:float(), target_size[2], target_size[1])
  -- else
  --   img = torch.randn(3, target_size[1], target_size[2]):float():mul(0.001)
  -- end

  -- if params.gpu >= 0 then
  --   if params.backend ~= 'clnn' then
  --     img = img:cuda()
  --   else
  --     img = img:cl()
  --   end
  -- end

  return net, style_losses, mask_net_avg, masks_modules
end
