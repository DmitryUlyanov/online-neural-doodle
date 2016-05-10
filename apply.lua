require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'image'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:text('Options:')
cmd:option('-mask_hdf5', '', 'Input mask')
cmd:option('-model', '', 'Input mask')
cmd:option('-out_path', '', 'Input mask')
cmd:text()

local params = cmd:parse(args or arg or {})

-- Load mask
local mask_file = hdf5.open(params.mask_hdf5, 'r')
local mask = mask_file:read('mask'):all()
mask_file:close()

-- Load model
local model = torch.load(params.model)
model:evaluate()
model:cuda()

local out = model:forward((mask:add_dummy()/10):cuda())
local im = torch.clamp(deprocess(out[1]:double()),0,1)

image.save(params.out_path, im)
