--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 6/21/14
-- Time: 11:06 PM
-- To change this template use File | Settings | File Templates.
--

local Tree = torch.class("Tree")

function bracketTokenizer(s, f)
	local pos = 1
	while pos < (#s+1) do
		local ch = s:sub(pos, pos)

		if ch == '(' then
			pos = pos + 1
			f(ch)
		elseif ch == ')' then
			pos = pos + 1
			f(ch)
		elseif ch == ' ' then
			pos = pos +  1
		else
			local start = pos
			repeat
				pos = pos + 1
				ch = s:sub(pos, pos)
			until pos >= (#s+1) or ch == ' ' or ch == '(' or ch == ')'
			f(s:sub(start, pos-1))
		end
	end
	f(nil)
end

function parseTree(tokenizer, createFn, dict)
	repeat
		tok = tokenizer()
		if tok == '(' then
			local name = tokenizer()
			local tree = createFn(name)
			tree.value = dict[name]

			repeat
				local child = parseTree(tokenizer, createFn, dict)
				if child ~= nil then
					tree:addChild(child)
				end
			until child == nil
			return tree
		elseif tok == ')' then
			return nil
		else
			local tree = createFn(tok)
			tree.value = dict[tok]
			return tree
		end
	until tok == nil
end

function Tree:__init(name)
	self.name = name
	self.value = nil
	self.children = {}
	self.parent = nil
end

function Tree:addChild(child)
	table.insert(self.children, child)
	child.parent = self
end

function Tree:isLeaf(child)
	return #self.children == 0
end

function Tree:filter(fn)
	local results = {fn(self)}
	for _, child in ipairs(self.children) do
		local subtree = child:filter(fn)
		for k, v in ipairs(subtree) do
			table.insert(results, v)
		end
	end

	return results
end

function Tree:apply(fn)
	fn(self)
	for _, child in ipairs(self.children) do
		child:apply(fn)
	end
end

function Tree:copy()
	local treecopy = Tree.new(self.name)

	for i, child in ipairs(self.children) do
		local subcopy = child:copy()
		treecopy:addChild(subcopy)
	end

	return treecopy
end

function Tree.parse(s, dict)
	local tokenizer = coroutine.wrap(function () bracketTokenizer(s, coroutine.yield) end)
	dict = dict or {}
	return parseTree(tokenizer, Tree.new, dict)
end

function Tree:describe()
	local result = nil

	result = "(" .. self.name

	for i, child in ipairs(self.children) do
		if child:isLeaf() then
			result = result .. " " .. child.name
		else
			result = result .. " " .. child:describe()
		end
	end

	result = result .. ")"

	return result
end

function Tree:__tostring__()
	return self:describe()
end


local s = "( all hippo ) bark"
local tree = Tree.parse(s)
print(tree)
