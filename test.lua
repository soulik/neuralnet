local nl = require 'utils/nl'
local n = require 'ai/neural'

local X = nl.array {
	{0,0,1},
	{0,1,1},
	{1,0,1},
	{1,1,1}
}

local y = nl.array {{0,0,1,1}}

local N1 = n.new(X, y, 2)

for i=1,10000 do
	N1.step()
	if i%100==0 then
		print(N1.error)
	end
end

print(N1.layer[2])