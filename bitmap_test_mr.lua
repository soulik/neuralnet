local nl = require 'utils/nl'
local n = require 'ai/neural'

local function nnet(layersCount)
	local layersCount = layersCount or 2
	local obj = {}
	local X,y = {}, {}
	local N = n.new(layersCount)

	obj.learn = function(iterations)
		local iterations = iterations or 2000
		local progress = n.progress(80, iterations)
		N.init(nl.array(X), nl.array(y))
		N.load('font2.dat')

		for i=1,iterations do
			N.step()
			progress(i)
		end

		print(('Last error: %0.4f%%'):format((N.error)*100))
		N.save('font2.dat')
		--print(N.layer[layersCount])
	end

	obj.resolve = function(x)
		N.load('font2.dat')
		return N.propagate(nl.array(x))
	end

	local function addBitmap(data, result)
		table.insert(X, data)
		table.insert(y, result)
	end

	obj.add = function(t)
		assert(type(t)=='table')
		for k, v in pairs(t) do
			if type(v)=='number' then
				addBitmap(k, {v})
			else
				addBitmap(k, v)
			end
		end
	end

	return obj
end

local l = nnet(4)
l.add({
	[{
		0,0,1,0,0,
		0,1,0,1,0,	
		1,0,0,0,1,	
		1,1,1,1,1,	
		1,0,0,0,1,	
		1,0,0,0,1,	
	
	}] = {0,0,1},
	[{
		1,1,1,1,0,
		1,0,0,1,1,	
		1,1,1,1,0,	
		1,0,0,0,1,	
		1,0,0,0,1,	
		1,1,1,1,0,	
	
	}] = {0,1,0},
	[{
		0,1,1,1,0,
		1,0,0,0,1,	
		1,0,0,0,0,	
		1,0,0,0,0,	
		1,0,0,0,1,	
		0,1,1,1,0,	
	
	}] = {0,1,1},
})
l.learn(10000)

local result = l.resolve {
		1,1,1,1,0,
		1,0,0,1,1,	
		1,1,1,1,0,	
		1,0,0,0,1,	
		1,0,0,0,1,	
		1,1,1,1,0,
}
print(result)
--[[
print(N1.propagate(nl.array {
	{1,0,1},
}))
--]]