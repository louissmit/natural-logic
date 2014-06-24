--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 6/22/14
-- Time: 3:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
utils = {}

function utils.split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
	 table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

function utils.getRandomVector(size, stop, start)
     return torch.Tensor(size):apply(function(x) return torch.uniform(start, stop) end)
end

function utils.getVocab(vectorSize)
    local vocab_file = assert(io.open('vector-entailment/wordpairs-v2.tsv', "r"))
    local vocab = {}
    repeat
        local sent = vocab_file:read()
        if sent ~= nil then
            local words = utils.split(sent, '\t')
            if #words == 3 then
                for i = 2,3 do
                    vocab[words[i]] = utils.getRandomVector(vectorSize, -0.01 , 0.01)
                end
            end
        end

    until sent == nil
    return vocab
end

return utils