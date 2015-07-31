local serpent = require 'serpent'
local nl = require 'utils/nl'

math.randomseed(os.time())

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

local function test(X,y, layers)
	local layers = layers or 2
	local function loadData(data)
		r, t = assert(serpent.load(data))
		return t
	end

	local function saveData(t)
		return serpent.dump(t)
	end

	local net = {
		syn = {},
		l = {},
	}

	setmetatable(net, {
		__index = {
			save = function()
				local t = {
					syn = {},
					l = {},
				}
				for i,v in ipairs(net.syn) do
					t.syn[i] = v.table
				end
				for i,v in ipairs(net.l) do
					t.l[i] = v.table
				end

				assert(io.open('network.dat','w')):write(saveData(t)):close()
			end,
			load = function()
				local f = io.open('network.dat','r')
				if f then
					local t0 = loadData(f:read('*a'))
					for i,v in ipairs(t0.syn) do
						net.syn[i] = nl.array(v)
					end
					for i,v in ipairs(t0.l) do
						net.l[i] = nl.array(v)
					end
				else
					for i=1,layers do
						local dim
						if i==1 then
							dim = {X.dim.cols, X.dim.rows}
						elseif i==layers then
							dim = {y.dim.rows, y.dim.cols}
						else
							dim = {X.dim.rows, X.dim.rows}
						end

						net.syn[i] = 2 * nl.random.random(dim) - 1
		        	end
				end
			end,
		},
	})

	local steps = 10000

	local syn = net.syn
	local layer = net.l

	net.load()
	
	layer[0] = X

	local p = progress(80, steps)

	for j=1,steps do
		local layerDelta = {}
		for l=1,layers do
			layer[l] = 1 / (1 + (-layer[l-1].dot(syn[l])).exp)
	    end

		for l=layers,1,-1 do
			if (l == layers) then
				layerDelta[l] = (y - layer[l]).elmProd(layer[l].elmProd(1 - layer[l]))
			else
				layerDelta[l] = layerDelta[l+1].dot(syn[l+1].T).elmProd(layer[l].elmProd(1 - layer[l]))
			end
		end

		for l=layers,1,-1 do
			if l==1 then
				syn[l].add(X.T.dot(layerDelta[l]))
			else
				syn[l].add(layer[l-1].T.dot(layerDelta[l]))
			end			
		end

		p(j)
	end
	net.save()

	print(syn[1])
	print(syn[layers])
end

local X = nl.array {{0,0,1},{0,1,1},{1,0,1},{1,1,1}}
local y = nl.array {{0,1,1,0}}.T

test(X,y,2)