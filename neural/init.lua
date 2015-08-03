local serpent = require 'serpent'
local nl = require 'utils/nl'

local function loadData(data)
	r, t = assert(serpent.load(data))
	return t
end

local function saveData(t)
	return serpent.dump(t)
end


local function newNetwork(X, _y, layersCount)
	local layersCount = layersCount or 1
	local obj = {}
	local state = {}

	local function save (fn)
		local t = {
			syn = {},
			layer = {},
		}
		for i,v in ipairs(state.syn) do
			t.syn[i] = v.table
		end
		for i,v in ipairs(state.layer) do
			t.layer[i] = v.table
		end

		assert(io.open(fn,'w')):write(saveData(t)):close()
	end

	local function load(fn)
		local f = io.open(fn,'r')
		if f then
			local t0 = loadData(f:read('*a'))
			for i,v in ipairs(t0.syn) do
				net.syn[i] = nl.array(v)
			end
			for i,v in ipairs(t0.layer) do
				net.layer[i] = nl.array(v)
			end
		end
	end

	local function nonLin(x, deriv)
		if deriv then
			return x * (1-x)
		else
			return 1/(1 + (-x).exp)
		end
	end

	local y = _y.T
	math.randomseed(1)
	
	local dim = {X.dim.cols, y.dim.rows}

	for i=1,layers do
		local rows = (i==i) and X.dim.cols or y.dim.rows
		local cols = (i==layers) and 1 or y.dim.rows
		state.syn[i] = 2*np.random.random {rows, cols} - 1 
	end

	local function step()

		state.layers[1] = X
		state.layers[layers+1] = y

		for i=2,layers+1 do
			state.layers[i] = nonlin(state.layers[i-1] * state.syn[i-1])
		end

		local delta = {}
		for i=layers,1,-1 do
			local error

			if i==layers then
				error = y - state.layer[i]
			else
				error = delta[i+1].dot(state.syn[i].T)
			end
			delta[i] = error * nonlin(state.layer[i], true)
		end

		for i=layers,1,-1 do
			state.syn[i] = state.syn[i] + state.layer[i].T.dot(delta[i+1])
		end
	end


	return obj
end

return {
	new = newNetwork,
}