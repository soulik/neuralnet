local serpent = require 'serpent'
local nl = require 'utils/nl'

local function loadData(data)
	r, t = assert(serpent.load(data))
	return t
end

local function saveData(t)
	return serpent.dump(t)
end

local function drawProgress(percent, elms, firstTime)
	local fill = math.floor(elms * (percent/100))
	local empty = elms - fill
	local strOut = ('[%s%s] %d%%'):format(('='):rep(fill), (' '):rep(empty), percent)
	if not firstTime then
		io.write(("\r"):rep(#strOut))
	end
	io.write(strOut)
end

local function progress(elms,steps)
	local firstTime = true
	local lastPercentage = -1
	local time0 = os.clock()

	return function(j)
		local percentDone = math.floor(100*j/steps)

		if (percentDone ~= lastPercentage) then
			lastPercentage = percentDone
			drawProgress(percentDone, elms, firstTime)
			firstTime = false
			if percentDone > 99 then
				local time1 = os.clock() - time0
				io.write(("\n(Total time: %0.3fs with %0.3fms per cycle)\n\n"):format(time1, (time1/steps)*1000))
			end
		end		
	end
end

local function newNetwork(X, _y, layersCount)
	local layersCount = layersCount or 1
	local obj = {}
	local state = {
		syn = {},
		layer = {},
	}

	local function save (fn)
		local t = {
			syn = {},
		}
		for i=0,layersCount-1 do
			t.syn[i] = state.syn[i].table
		end

		assert(io.open(fn,'w')):write(saveData(t)):close()
	end

	local function load(fn)
		local f = io.open(fn,'r')
		if f then
			local t0 = loadData(f:read('*a'))
			for i=0,layersCount-1 do
				state.syn[i] = nl.array(t0.syn[i])
			end
		end
	end

	local function nonlin(x, deriv)
		if deriv then
			return x.elmProd (1-x)
		else
			return 1 / (1 + (-x).exp)
		end
	end

	local y = _y.T
	math.randomseed(1)
	
	for i=0,layersCount-1 do
		local rows = (i==0) and X.dim.cols or y.dim.rows
		local cols = (i==layersCount-1) and 1 or y.dim.rows
		state.syn[i] = 2*nl.random.random {rows, cols} - 1
	end

	local function propagate(X)
		for i=0,layersCount do
			if i==0 then
				state.layer[i] = X
			else
				state.layer[i] = nonlin(state.layer[i-1] * state.syn[i-1])
			end
		end
		return state.layer[layersCount]
	end

	local function step()

		for i=0,layersCount do
			if i==0 then
				state.layer[i] = X
			else
				state.layer[i] = nonlin(state.layer[i-1].dot(state.syn[i-1]))
			end
		end

		local delta = {}
		for i=layersCount,1,-1 do
			local error

			if i==layersCount then
				error = y - state.layer[i]
				obj.error = error.abs.mean
			else
				error = delta[i+1].dot(state.syn[i].T)	
			end

			if type(error)=='number' then
				delta[i] = error * nonlin(state.layer[i], true)
			else				
				delta[i] = error.elmProd(nonlin(state.layer[i], true))
			end
		end

		for i=layersCount-1,0,-1 do
			state.syn[i].add(state.layer[i].T.dot(delta[i+1]))
		end
	end

	obj.step = step
	obj.propagate = propagate
	obj.load = load
	obj.save = save
	obj.layer = state.layer

	return obj
end

return {
	new = newNetwork,
	progress = progress,
}