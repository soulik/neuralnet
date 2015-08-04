local nl = require 'utils/nl'
local n = require 'ai/neural'

local X = nl.array {
	{0,0,1},
	{0,1,1},
	{1,0,1},
	{1,1,1}
}

local y = nl.array {{0,1,1,0}}

local N1 = n.new(X, y, 2)
N1.load('neural.dat')

local iterations = 20000 
local progress = n.progress(80, iterations)

for i=1,iterations do
	N1.step()
	progress(i)
	--if i%1000==0 then
	--end
end
print(('Last error: %0.4f%%'):format((N1.error)*100))

N1.save('neural.dat')

print(N1.layer[2])
