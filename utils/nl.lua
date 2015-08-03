local M = {
	random = {},
}

M.array = function(v)
	assert(type(v)=='table', 'Expected table')
	if type(v[1])~='table' then
		v = {v}
	end
	local obj = {}
	local dim = {rows=0,cols=0}
	dim.rows = #v
	if dim.rows >= 1 then
		dim.cols = #(v[1])
	end
	setmetatable(dim, {
		__eq = function(a, b)
			return (a.rows == b.rows and a.cols == b.cols)
		end,
		__tostring = function(t)
			return ('%dx%d'):format(dim.rows, dim.cols)
		end,
	})

	local iteratorRC = function()
		return coroutine.wrap(function()
			for i=1,dim.rows do
				for j=1,dim.cols do
					coroutine.yield(i, j, v[i][j])
				end
			end
		end)
	end

	local iteratorCR = function()
		return coroutine.wrap(function()
			for j=1,dim.cols do
				for i=1,dim.rows do
					coroutine.yield(i, j, v[i][j])
				end
			end
		end)
	end

	local function transpose()
		local out = {}
		for j=1,dim.cols do
			local col = out[j]
			if type(col)~='table' then
				col = {}; out[j] = col
			end
			for i=1,dim.rows do
				col[i] = v[i][j]
			end
		end
		return M.array(out)
	end

	local function multiplyMatrices(a, b)
		local out = {}
		assert(a.dim.cols == b.dim.rows, ('Invalid matrix dimensions (%s %s)'):format(tostring(a.dim), tostring(b.dim)))

		for i = 1, a.dim.rows do
			local row = out[i]
			if type(row)~='table' then
				row = {}; out[i] = row
			end
			for j = 1, b.dim.cols do
				local sum = 0
				for k = 1, a.dim.cols do
					sum = sum + a{i, k} * b{k, j}
				end
				row[j] = sum
			end
		end

		return out
	end

	local function dotProductVectors(a, b)
		assert(type(a)=='table' and type(b)=='table', 'Vector dot product can be used only on vectors')
		assert(a.dim.rows == b.dim.rows and a.dim.cols==b.dim.cols, ('Vectors must have the same size (%s %s)'):format(tostring(a.dim), tostring(b.dim)))
		local sum = 0

		for i = 1, a.dim.cols do
			sum = sum + a{1, i} * b{1, i}
		end

		return sum
	end
	
	local function dotProductML(a, b, c)
		assert(type(a)=='table' and type(b)=='table', 'Dot product can be used only on matrices and vectors')
		assert(a.dim.rows == b.dim.rows and a.dim.cols==b.dim.cols, ('Matrices or vectors must have the same size (%s %s)'):format(tostring(a.dim), tostring(b.dim)))

		if a.dim.rows == 1 then
			return dotProductVectors(a, b)
		else
			local out = {}
			if c == 2 then
				for i = 1 , a.dim.rows do
					out[i] = {dotProductVectors(a.row(i), b.row(i))}
				end
			else
				local row = {}; out[1] = row
				for j = 1 , a.dim.cols do
					row[j] = dotProductVectors(a.col(j), b.col(j))
				end
			end
			return M.array(out)
	    end
	end

	local function dotProductMatrix(a, b)
		assert(type(a)=='table' and type(b)=='table', 'Dot product can be used only on matrices and vectors')
		--assert(a.dim.rows == b.dim.rows and a.dim.cols==b.dim.cols, ('Matrices or vectors must have the same size (%s %s)'):format(tostring(a.dim), tostring(b.dim)))

		if a.dim.rows == 1 then
			return dotProductVectors(a, b)
		else
			--[[
			local out = {}
			for i= 1 , a.dim.rows do
				local row = out[i]
				if type(row)~='table' then
					row = {}; out[i] = row
				end
				for j = 1, a.dim.cols do
					row[j] = (row[j] or 0) + dotProductVectors(a.row(i), b.col(j))
				end
			end
			--]]
			return a * b
	    end
	end

	local function row(i)
		local row = {}
		for j=1,dim.cols do
			row[j] = obj {i, j}
		end
		return M.array(row)
	end

	local function col(j)
		local col = {}
		for i=1,dim.rows do
			col[i] = obj {i, j}
		end
		return M.array(col)
	end

	local function tostring()
		local out = {}
		table.insert(out, ("Matrix: %dx%d\n"):format(dim.rows, dim.cols))
		for i=1,dim.rows do
			local row = v[i]
			for j=1,dim.cols do
				table.insert(out, ("%03f "):format(tonumber(row[j])))
	    	end
			table.insert(out, "\n")
		end
		return table.concat(out)
	end

	local function getVal(row, col)
		assert(row >= 1 and row <= dim.rows, 'Invalid row')
		assert(col >= 1 and col <= dim.cols, 'Invalid column')
		return v[row][col]
	end

	local function setVal(row, col, value)
		assert(row >= 1 and row <= dim.rows, 'Invalid row')
		assert(col >= 1 and col <= dim.cols, 'Invalid column')
		v[row][col] = value
	end

	local function symmetricBinaryOperator(a, b, fn, fnTT)
		local out = {}
		if type(a)=='table' and type(b)=='number' then
			for i,j in iteratorRC() do
				local row = out[i]
				if type(row)~='table' then
					row = {}; out[i] = row
				end
				row[j] = fn(a{i, j}, b)
			end
		elseif type(a)=='number' and type(b)=='table' then
			for i,j in iteratorRC() do
				local row = out[i]
				if type(row)~='table' then
					row = {}; out[i] = row
				end
				row[j] = fn(a, b{i, j})
			end
		elseif type(a)=='table' and type(b)=='table' then
			if type(fnTT)=='function' then
				out = fnTT(a, b)
			else
				assert(a.dim.rows == b.dim.rows and a.dim.cols == b.dim.cols, ('Matrices must have the same size (%s %s)'):format(tostring(a.dim), tostring(b.dim)))
				for i,j in iteratorRC() do
					local row = out[i]
					if type(row)~='table' then
						row = {}; out[i] = row
					end
					row[j] = fn(a{i, j}, b{i, j})
				end
			end
		end
		return M.array(out)
	end

	local function symmetricInplaceBinaryOperator(b, fn)
		if type(b)=='table' then
			assert(dim.rows == b.dim.rows and dim.cols == b.dim.cols, ('Matrices must have the same size (%s %s)'):format(tostring(dim), tostring(b.dim)))
			for i,j,v in iteratorRC() do
				obj[{i,j}] = fn(v, b{i, j})
			end
		end
	end

	local function unaryOperator(a, fn)
		local out = {}
		if type(a)=='table' then
			for i,j in iteratorRC() do
				local row = out[i]
				if type(row)~='table' then
					row = {}; out[i] = row
				end
				row[j] = fn(a{i, j})
			end
		end
		return M.array(out)
	end

	local getters = {
		T = transpose,
		iteratorRC = iteratorRC,
		iteratorCR = iteratorCR,
		dim = dim,
		sum = function()
			local sum = 0
			for _,_,v in iteratorRC() do
				sum = sum + v
			end
			return sum
		end,
		table = function()
			return v
		end,
	}

	local special = {
		elmProd = function(b)
			return symmetricBinaryOperator(obj, b, function(a,b) return a * b end)
		end,
		dot = function(...)
			return dotProductMatrix(obj, ...)
		end,
		dot2 = function(...)
			return dotProductML(obj, ...)
		end,
		row = row,
		col = col,
		add = function(b)
			symmetricInplaceBinaryOperator(b, function(a,b) return a + b end)
		end,
		sub = function(b)
			symmetricInplaceBinaryOperator(b, function(a,b) return a - b end)
		end,
		mul = function(b)
			symmetricInplaceBinaryOperator(b, function(a,b) return a * b end)
		end,
		div = function(b)
			symmetricInplaceBinaryOperator(b, function(a,b) return a / b end)
		end,
	}

	for i, fnName in ipairs {
		'abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'deg', 'exp', 'floor', 'frexp', 'log', 'log10', 'rad', 'sin', 'sinh', 'sqrt', 'tan', 'tanh'
	} do
		getters[fnName] = function()
			return unaryOperator(obj, function(a) return math[fnName](a) end)
		end
	end

	setmetatable(obj, {
		__index = function(t, k)
			if type(k)=='table' then
				assert(#k == 2, 'Expected two coordinates')
				return getVal(k[1], k[2])
			elseif type(k)=='string' then
				local g = getters[k]
				if type(g)=='function' then
					return g()
				elseif (type(g) ~= 'nil') then
					return g
				else
					return special[k]
				end
			end
		end,
		__newindex = function(t, k, v)
			if type(k)=='table' then
				assert(#k == 2, 'Expected two coordinates')
				setVal(k[1], k[2], v)
			end
		end,
		__call = function(t, k)
			if type(k)=='table' then
				assert(#k == 2, 'Expected two coordinates')
				return getVal(k[1], k[2])
			end
		end,
		__add = function(a, b)
			return symmetricBinaryOperator(a,b, function(a,b) return a+b end)
		end,
		__sub = function(a, b)
			return symmetricBinaryOperator(a,b, function(a,b) return a - b end)
		end,
		__unm = function(a)
			return unaryOperator(a, function(a) return -a end)
		end,
		__len = function(a)
			return dim
		end,
		__mul = function(a, b)
			return symmetricBinaryOperator(a, b, function(a,b) return a * b end, multiplyMatrices)
		end,
		__div = function(a, b)
			return symmetricBinaryOperator(a, b, function(a,b) return a / b end)
		end,
		__pow = function(a, b)
			return symmetricBinaryOperator(a, b, function(a,b) return a^b end)
		end,
		__mod = function(a, b)
			return symmetricBinaryOperator(a, b, function(a,b) return a % b end)
		end,
		__tostring = tostring,
	})
	return obj
end

M.random.random = function(size)
	local dim = {rows=0,cols=0}
	if type(size)=='number' then
		dim.rows = size
		dim.cols = 1
	elseif type(size)=='table' then
		assert(#size == 2, 'Expected two coordinates')
		dim.rows = size[1]
		dim.cols = size[2]
	end

	local out = {}
	for i=1,dim.rows do
		local row = out[i]
		if type(row)~='table' then
			row = {}; out[i] = row
		end
		for j=1,dim.cols do
			row[j] = math.random()
		end
	end
	return M.array(out)
end

return M