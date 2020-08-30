#!/usr/bin/env luajit

require 'torch'

csv2tensor = require 'csv2tensor'    -- luarocks install csv2tensor 

bupa_tensor, column_names = csv2tensor.load("bupa_liver_disorders.csv") 

for col_idx, col_name in pairs(column_names)
do
    print("Column " .. col_idx .. ": " .. col_name)
end

print(bupa_tensor)

