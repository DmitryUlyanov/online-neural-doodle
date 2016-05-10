local MergeLastDims, parent = torch.class('nn.MergeLastDims', 'nn.Module')

function MergeLastDims:__init()
   parent.__init(self)
end

function MergeLastDims:updateOutput(input)
  -- Assume input is 4D
  -- B x C x W x H
  local sz = input:size()
  -- B x C x WH
  self.output = input:view(sz[1],sz[2],sz[3]*sz[4])

  return self.output
end

function MergeLastDims:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:viewAs(input)
   return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.MergeLastDims())
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target_grams, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target_grams = target_grams
  self.loss = 0
  
  -- B x M x W x H
  self.target_masks = nil

  self.gram = GramMatrix():cuda()
  -- self.crit = nn.SmoothL1Criterion()
  self.crit = nn.MSECriterion():cuda()

  self.gradInput = nil
end

function StyleLoss:updateOutput(input)
  -- We do everything in updateGradInput to save memory
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)

  self.gradInput = self.gradInput or gradOutput:clone()
  self.gradInput:zero()

  self.loss = 0 

  -- B x M 
  local mask_norms = (self.target_masks:mean(3):mean(4) + 1e-6) * input[1]:nElement()

  -- Apply masks
  for k = 1, self.target_masks:size(2) do

    -- B x M x W x H -> B x 1 x W x H
    local this_mask = self.target_masks:narrow(2,k,1)

    -- B x 1 x 1
    local this_mask_norm = mask_norms:narrow(2,k,1):reshape(mask_norms:size(1), 1, 1)
    
    -- B x C x W x H
    this_mask = this_mask:expandAs(input)

    -- Forward
    local masked_input = torch.cmul(input, this_mask)
    -- B x C x C
    local G = self.gram:forward(masked_input)

    -- Normalize
    G:cdiv(this_mask_norm:expandAs(G))

    local target_gram_exp = self.target_grams[k]:expandAs(G)
    self.loss = self.loss + self.crit:forward(G, target_gram_exp)  

    -- Backward
    local dG = self.crit:backward(G, target_gram_exp)
    
    -- Normalize grad
    dG:cdiv(this_mask_norm:expandAs(G))

    local gradInput = self.gram:backward(masked_input, dG)

    gradInput:cmul(this_mask)

    if self.normalize then
      gradInput:div(torch.norm(gradInput, 1) + 1e-8)
    end

    self.gradInput:add(gradInput)

  end

  self.gradInput:add(gradOutput)
  return self.gradInput
end