import { useState, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, CartesianGrid, Legend, Area, AreaChart, ComposedChart, ReferenceLine } from "recharts";

const DATA = {"VELVETFRUIT_EXTRACT":[[0,1,5245.0,6],[10000,1,5245.5,3],[20000,1,5227.0,6],[30000,1,5240.5,3],[40000,1,5234.0,6],[50000,1,5247.5,5],[60000,1,5246.0,6],[70000,1,5243.5,5],[80000,1,5237.5,5],[90000,1,5242.5,5],[100000,1,5248.5,5],[110000,1,5270.0,6],[120000,1,5258.5,5],[130000,1,5261.5,5],[140000,1,5258.5,5],[150000,1,5267.0,6],[160000,1,5253.5,5],[170000,1,5250.5,5],[180000,1,5245.0,6],[190000,1,5238.5,5],[200000,1,5238.5,5],[210000,1,5251.5,5],[220000,1,5248.5,5],[230000,1,5245.5,5],[240000,1,5254.5,5],[250000,1,5266.5,5],[260000,1,5253.0,6],[270000,1,5253.5,5],[280000,1,5260.0,6],[290000,1,5271.5,5],[300000,1,5268.5,5],[310000,1,5267.5,5],[320000,1,5243.0,6],[330000,1,5246.5,5],[340000,1,5250.0,6],[350000,1,5249.0,6],[360000,1,5247.5,5],[370000,1,5253.5,5],[380000,1,5251.0,6],[390000,1,5243.5,5],[400000,1,5241.5,5],[410000,1,5235.0,6],[420000,1,5228.0,6],[430000,1,5230.5,5],[440000,1,5235.5,5],[450000,1,5245.5,5],[460000,1,5250.5,3],[470000,1,5254.0,6],[480000,1,5251.0,6],[490000,1,5242.5,5],[500000,1,5243.5,5],[510000,1,5261.0,2],[520000,1,5260.5,3],[530000,1,5250.0,2],[540000,1,5236.5,5],[550000,1,5234.0,6],[560000,1,5230.5,5],[570000,1,5236.5,5],[580000,1,5234.5,5],[590000,1,5244.0,6],[600000,1,5234.5,5],[610000,1,5223.5,5],[620000,1,5234.5,5],[630000,1,5220.0,6],[640000,1,5213.5,5],[650000,1,5201.5,5],[660000,1,5211.5,5],[670000,1,5219.5,5],[680000,1,5224.5,5],[690000,1,5237.5,5],[700000,1,5238.5,5],[710000,1,5254.0,6],[720000,1,5266.5,5],[730000,1,5266.5,5],[740000,1,5264.5,5],[750000,1,5272.5,5],[760000,1,5276.5,5],[770000,1,5267.5,5],[780000,1,5259.0,6],[790000,1,5266.5,5],[800000,1,5250.5,5],[810000,1,5243.0,6],[820000,1,5236.0,6],[830000,1,5239.0,2],[840000,1,5254.5,5],[850000,1,5263.5,5],[860000,1,5262.5,5],[870000,1,5246.5,5],[880000,1,5239.0,6],[890000,1,5245.5,3],[900000,1,5249.5,5],[910000,1,5260.5,5],[920000,1,5270.5,5],[930000,1,5273.5,5],[940000,1,5262.0,6],[950000,1,5252.0,6],[960000,1,5262.5,5],[970000,1,5265.5,5],[980000,1,5259.5,5],[990000,1,5256.5,5],[0,2,5267.5,5],[10000,2,5262.5,5],[20000,2,5269.5,5],[30000,2,5264.5,5],[40000,2,5261.5,5],[50000,2,5260.5,5],[60000,2,5243.5,5],[70000,2,5257.5,5],[80000,2,5259.5,5],[90000,2,5268.5,5],[100000,2,5263.5,5],[110000,2,5261.5,5],[120000,2,5253.0,6],[130000,2,5252.5,3],[140000,2,5256.5,5],[150000,2,5267.5,5],[160000,2,5274.5,5],[170000,2,5267.5,1],[180000,2,5268.5,5],[190000,2,5260.5,5],[200000,2,5262.0,6],[210000,2,5267.5,5],[220000,2,5264.5,5],[230000,2,5265.0,6],[240000,2,5264.5,5],[250000,2,5277.5,5],[260000,2,5278.0,6],[270000,2,5278.0,6],[280000,2,5267.5,5],[290000,2,5272.5,5],[300000,2,5270.5,5],[310000,2,5273.0,6],[320000,2,5280.0,6],[330000,2,5262.0,6],[340000,2,5267.0,6],[350000,2,5262.0,6],[360000,2,5264.5,5],[370000,2,5252.5,5],[380000,2,5251.5,5],[390000,2,5239.5,5],[400000,2,5238.0,6],[410000,2,5227.5,5],[420000,2,5231.5,5],[430000,2,5236.5,5],[440000,2,5238.5,5],[450000,2,5249.5,5],[460000,2,5238.5,5],[470000,2,5237.0,6],[480000,2,5243.5,5],[490000,2,5239.0,6],[500000,2,5245.5,5],[510000,2,5235.5,5],[520000,2,5245.5,5],[530000,2,5248.5,3],[540000,2,5241.5,5],[550000,2,5234.5,5],[560000,2,5239.5,5],[570000,2,5229.5,5],[580000,2,5222.0,6],[590000,2,5213.5,5],[600000,2,5242.0,6],[610000,2,5251.5,5],[620000,2,5237.5,5],[630000,2,5245.0,6],[640000,2,5246.5,5],[650000,2,5242.5,5],[660000,2,5257.5,5],[670000,2,5265.5,5],[680000,2,5276.0,6],[690000,2,5274.5,5],[700000,2,5269.5,5],[710000,2,5257.5,5],[720000,2,5242.5,5],[730000,2,5224.5,5],[740000,2,5213.5,5],[750000,2,5215.5,5],[760000,2,5217.5,5],[770000,2,5237.5,5],[780000,2,5242.5,5],[790000,2,5241.5,5],[800000,2,5249.5,5],[810000,2,5254.5,3],[820000,2,5251.5,5],[830000,2,5251.5,5],[840000,2,5256.5,5],[850000,2,5255.5,5],[860000,2,5256.5,5],[870000,2,5246.5,5],[880000,2,5252.5,5],[890000,2,5265.5,5],[900000,2,5286.5,5],[910000,2,5276.5,5],[920000,2,5282.5,5],[930000,2,5276.0,6],[940000,2,5273.5,5],[950000,2,5274.5,5],[960000,2,5270.5,5],[970000,2,5262.5,5],[980000,2,5273.0,2],[990000,2,5289.5,5],[0,3,5295.5,5],[10000,3,5296.5,5],[20000,3,5276.5,5],[30000,3,5267.0,2],[40000,3,5260.5,5],[50000,3,5253.0,6],[60000,3,5252.5,5],[70000,3,5257.0,6],[80000,3,5260.0,6],[90000,3,5248.5,5],[100000,3,5252.5,5],[110000,3,5251.0,6],[120000,3,5239.5,5],[130000,3,5242.5,5],[140000,3,5245.5,5],[150000,3,5243.5,5],[160000,3,5233.5,5],[170000,3,5244.5,5],[180000,3,5247.0,6],[190000,3,5249.5,5],[200000,3,5238.5,5],[210000,3,5237.5,5],[220000,3,5233.5,5],[230000,3,5244.5,5],[240000,3,5231.0,2],[250000,3,5237.5,5],[260000,3,5231.5,5],[270000,3,5239.5,5],[280000,3,5236.5,5],[290000,3,5233.5,5],[300000,3,5223.5,5],[310000,3,5222.0,6],[320000,3,5223.5,5],[330000,3,5238.0,6],[340000,3,5230.0,6],[350000,3,5225.0,6],[360000,3,5216.5,5],[370000,3,5227.5,5],[380000,3,5244.5,5],[390000,3,5246.5,5],[400000,3,5254.5,5],[410000,3,5233.0,6],[420000,3,5229.5,5],[430000,3,5214.5,5],[440000,3,5216.5,5],[450000,3,5215.5,5],[460000,3,5214.5,5],[470000,3,5211.5,5],[480000,3,5216.5,5],[490000,3,5213.5,5],[500000,3,5201.5,5],[510000,3,5208.5,5],[520000,3,5204.5,5],[530000,3,5202.5,3],[540000,3,5200.0,6],[550000,3,5212.5,5],[560000,3,5216.5,5],[570000,3,5225.5,5],[580000,3,5219.5,5],[590000,3,5226.5,5],[600000,3,5220.5,5],[610000,3,5220.5,5],[620000,3,5224.5,5],[630000,3,5228.0,6],[640000,3,5237.0,6],[650000,3,5252.5,5],[660000,3,5264.5,5],[670000,3,5251.5,5],[680000,3,5254.5,5],[690000,3,5240.5,5],[700000,3,5247.5,5],[710000,3,5256.5,5],[720000,3,5257.0,2],[730000,3,5258.0,2],[740000,3,5243.0,6],[750000,3,5234.5,5],[760000,3,5229.0,6],[770000,3,5232.5,5],[780000,3,5233.5,5],[790000,3,5247.0,2],[800000,3,5245.5,5],[810000,3,5261.0,6],[820000,3,5251.5,5],[830000,3,5244.5,3],[840000,3,5240.5,5],[850000,3,5244.5,5],[860000,3,5237.0,6],[870000,3,5234.5,5],[880000,3,5228.0,2],[890000,3,5235.0,2],[900000,3,5251.5,5],[910000,3,5254.5,5],[920000,3,5258.0,6],[930000,3,5265.5,5],[940000,3,5254.5,5],[950000,3,5277.5,5],[960000,3,5271.5,5],[970000,3,5264.5,5],[980000,3,5244.5,5],[990000,3,5235.5,5]],"HYDROGEL_PACK":[[0,1,9958.0,16],[10000,1,9941.0,16],[20000,1,9951.0,16],[30000,1,9944.0,16],[40000,1,9945.0,16],[50000,1,9980.0,16],[60000,1,9959.0,16],[70000,1,9957.0,16],[80000,1,9984.0,16],[90000,1,9962.0,16],[100000,1,9945.0,16],[110000,1,9963.0,16],[120000,1,9955.0,16],[130000,1,9957.0,16],[140000,1,9963.0,16],[150000,1,9965.0,16],[160000,1,9970.0,16],[170000,1,9968.0,16],[180000,1,10002.0,16],[190000,1,10027.0,16],[200000,1,10014.0,16],[210000,1,10003.0,16],[220000,1,10003.0,16],[230000,1,9990.0,16],[240000,1,10011.0,16],[250000,1,10003.0,16],[260000,1,9979.0,16],[270000,1,9983.0,16],[280000,1,9963.5,7],[290000,1,9951.0,16],[300000,1,9960.0,16],[310000,1,9936.0,16],[320000,1,9948.0,16],[330000,1,9946.5,15],[340000,1,9941.0,16],[350000,1,9967.0,16],[360000,1,9956.0,16],[370000,1,9967.0,16],[380000,1,9950.0,16],[390000,1,9945.0,16],[400000,1,9941.0,8],[410000,1,9945.0,16],[420000,1,9947.0,16],[430000,1,9938.0,16],[440000,1,9919.0,16],[450000,1,9940.0,16],[460000,1,9924.0,16],[470000,1,9935.5,15],[480000,1,9957.0,16],[490000,1,9961.0,16],[500000,1,9999.0,16],[510000,1,10010.0,16],[520000,1,10013.0,16],[530000,1,10021.0,16],[540000,1,10019.0,16],[550000,1,10008.0,16],[560000,1,10001.0,16],[570000,1,10012.0,16],[580000,1,10003.0,16],[590000,1,9996.0,16],[600000,1,10024.0,16],[610000,1,10012.0,16],[620000,1,10002.0,16],[630000,1,10015.0,16],[640000,1,10029.0,16],[650000,1,9997.0,16],[660000,1,9996.0,16],[670000,1,10018.0,16],[680000,1,9998.0,16],[690000,1,10000.0,16],[700000,1,10050.5,9],[710000,1,10040.0,16],[720000,1,10061.0,16],[730000,1,10066.0,16],[740000,1,10044.0,16],[750000,1,10053.5,17],[760000,1,10058.0,16],[770000,1,10069.0,16],[780000,1,10070.0,16],[790000,1,10052.0,16],[800000,1,10015.5,7],[810000,1,10006.0,8],[820000,1,10007.0,16],[830000,1,10008.0,16],[840000,1,10027.0,16],[850000,1,10037.0,16],[860000,1,10078.0,16],[870000,1,10050.0,16],[880000,1,9994.0,16],[890000,1,10009.0,16],[900000,1,10000.0,16],[910000,1,10024.0,16],[920000,1,10011.0,16],[930000,1,10003.0,16],[940000,1,10003.0,16],[950000,1,10033.0,16],[960000,1,10022.0,16],[970000,1,9986.0,16],[980000,1,10013.0,16],[990000,1,10026.0,16],[0,2,10011.0,16],[10000,2,10023.0,16],[20000,2,10016.0,16],[30000,2,10005.0,16],[40000,2,9993.0,16],[50000,2,9952.0,16],[60000,2,9962.0,16],[70000,2,9996.0,16],[80000,2,9960.0,16],[90000,2,9927.0,16],[100000,2,9960.0,16],[110000,2,9980.0,16],[120000,2,9995.0,8],[130000,2,10002.0,16],[140000,2,10014.0,16],[150000,2,10016.0,16],[160000,2,10027.0,16],[170000,2,10012.0,16],[180000,2,10009.0,16],[190000,2,9974.0,16],[200000,2,9969.0,16],[210000,2,9975.0,16],[220000,2,9956.0,16],[230000,2,9911.0,16],[240000,2,9918.0,16],[250000,2,9937.5,15],[260000,2,9926.0,16],[270000,2,9924.0,16],[280000,2,9894.5,7],[290000,2,9921.0,16],[300000,2,9952.5,15],[310000,2,9936.0,16],[320000,2,9960.0,16],[330000,2,9949.5,15],[340000,2,9958.0,16],[350000,2,9955.0,16],[360000,2,9983.0,16],[370000,2,9985.0,16],[380000,2,9977.0,16],[390000,2,9981.0,16],[400000,2,9974.0,16],[410000,2,9980.0,16],[420000,2,9970.0,16],[430000,2,9961.0,16],[440000,2,9967.0,16],[450000,2,9986.0,16],[460000,2,9967.5,15],[470000,2,9990.0,16],[480000,2,10000.0,16],[490000,2,9995.0,16],[500000,2,9977.0,16],[510000,2,9997.0,16],[520000,2,9970.0,16],[530000,2,9992.0,16],[540000,2,9971.0,16],[550000,2,9966.0,16],[560000,2,9984.0,16],[570000,2,9988.0,16],[580000,2,9986.0,16],[590000,2,10008.0,16],[600000,2,10007.0,16],[610000,2,10036.0,16],[620000,2,10018.0,16],[630000,2,10024.0,16],[640000,2,10009.0,16],[650000,2,10037.0,16],[660000,2,10040.0,16],[670000,2,10029.0,16],[680000,2,9998.0,16],[690000,2,10007.0,16],[700000,2,10009.0,16],[710000,2,10032.5,9],[720000,2,10039.0,16],[730000,2,10035.0,16],[740000,2,10009.0,16],[750000,2,10005.0,16],[760000,2,10025.0,16],[770000,2,10034.0,16],[780000,2,10027.0,16],[790000,2,10003.0,16],[800000,2,9989.0,16],[810000,2,10000.0,16],[820000,2,10026.0,16],[830000,2,10034.0,16],[840000,2,10012.0,16],[850000,2,9999.0,16],[860000,2,10015.0,16],[870000,2,9997.0,16],[880000,2,9986.0,16],[890000,2,9989.0,16],[900000,2,10005.0,8],[910000,2,10000.0,16],[920000,2,10007.0,16],[930000,2,9998.0,16],[940000,2,10020.5,9],[950000,2,9987.0,16],[960000,2,10014.0,16],[970000,2,9997.0,16],[980000,2,9988.0,16],[990000,2,10013.0,16],[0,3,10008.0,16],[10000,3,9995.0,16],[20000,3,10012.0,16],[30000,3,10034.5,17],[40000,3,10030.0,16],[50000,3,10042.0,16],[60000,3,10056.0,16],[70000,3,10043.0,16],[80000,3,10044.0,16],[90000,3,10042.5,9],[100000,3,10014.0,16],[110000,3,10025.0,16],[120000,3,10031.0,16],[130000,3,10022.0,16],[140000,3,10016.0,16],[150000,3,10003.0,16],[160000,3,10037.0,16],[170000,3,10038.0,16],[180000,3,10034.0,16],[190000,3,10003.0,16],[200000,3,10019.0,16],[210000,3,9997.0,16],[220000,3,10020.0,16],[230000,3,9977.5,15],[240000,3,10021.0,16],[250000,3,10037.0,16],[260000,3,10012.0,16],[270000,3,9996.0,16],[280000,3,10008.0,16],[290000,3,10015.0,16],[300000,3,10015.0,16],[310000,3,9996.0,16],[320000,3,9979.5,7],[330000,3,9992.0,16],[340000,3,10016.0,16],[350000,3,10038.0,16],[360000,3,10031.0,16],[370000,3,10039.0,16],[380000,3,10025.0,16],[390000,3,10007.0,16],[400000,3,10019.0,16],[410000,3,9984.0,16],[420000,3,10020.0,16],[430000,3,10036.0,16],[440000,3,10039.0,16],[450000,3,10048.5,7],[460000,3,10078.0,16],[470000,3,10073.5,17],[480000,3,10033.0,16],[490000,3,10058.5,9],[500000,3,10040.0,16],[510000,3,10043.0,16],[520000,3,10059.0,16],[530000,3,10046.0,16],[540000,3,10005.0,16],[550000,3,10013.0,16],[560000,3,9996.0,16],[570000,3,9993.0,16],[580000,3,9978.0,16],[590000,3,10010.0,16],[600000,3,10023.0,16],[610000,3,10011.0,16],[620000,3,10016.0,16],[630000,3,9982.0,16],[640000,3,9974.0,16],[650000,3,9999.0,16],[660000,3,9954.0,16],[670000,3,9960.0,16],[680000,3,9959.0,16],[690000,3,9958.0,16],[700000,3,9971.0,16],[710000,3,9950.0,16],[720000,3,9953.0,16],[730000,3,9952.0,16],[740000,3,9953.0,16],[750000,3,9971.0,16],[760000,3,9970.0,16],[770000,3,9978.0,16],[780000,3,9964.5,15],[790000,3,9957.0,16],[800000,3,9963.0,16],[810000,3,9975.0,16],[820000,3,9965.0,16],[830000,3,9990.0,16],[840000,3,10029.5,17],[850000,3,10024.0,16],[860000,3,10007.0,16],[870000,3,10019.0,16],[880000,3,9970.0,16],[890000,3,9944.0,16],[900000,3,9957.0,16],[910000,3,9929.0,16],[920000,3,9934.0,16],[930000,3,9944.0,16],[940000,3,9971.0,16],[950000,3,9992.0,16],[960000,3,9968.0,16],[970000,3,9975.0,16],[980000,3,9964.0,16],[990000,3,9982.0,16]],"vfe_acf":[-0.1601,-0.011,-0.0032,0.0102,-0.0039,0.0007,-0.0009,0.0049,-0.0013,0.0028,0.0042,0.0063,-0.0063,0.0128,-0.0057,-0.0022,0.0086,-0.0002,0.0073,-0.007,-0.0062,-0.0045,0.0034,0.0023,-0.001,0.0126,-0.0104,-0.0005,0.0009,0.0032],"hp_acf":[-0.1242,0.0062,-0.0033,-0.0032,0.0027,0.0028,-0.0046,0.0034,0.0096,-0.0057,-0.003,-0.0055,-0.0118,0.0153,-0.0124,0.0021,0.0028,-0.0061,0.0002,0.0042,-0.0067,-0.0004,0.0022,0.0033,-0.0107,0.0023,-0.0039,-0.0096,0.0015,-0.0007],"stats":{"VELVETFRUIT_EXTRACT":{"m":5247.6,"s":18.1,"lo":5191.5,"hi":5300.0,"sp":4.98,"v":75},"HYDROGEL_PACK":{"m":9994.7,"s":34.6,"lo":9891.0,"hi":10081.0,"sp":15.73,"v":24},"VEV_6000":{"m":0.5,"s":0.0,"lo":0.5,"hi":0.5,"sp":1.0,"v":44},"VEV_5000":{"m":251.1,"s":17.5,"lo":196.5,"hi":301.5,"sp":5.96,"v":31},"VEV_6500":{"m":0.5,"s":0.0,"lo":0.5,"hi":0.5,"sp":1.0,"v":30},"VEV_5300":{"m":41.2,"s":9.1,"lo":18.0,"hi":61.0,"sp":1.97,"v":41},"VEV_5400":{"m":12.6,"s":4.1,"lo":3.5,"hi":23.0,"sp":1.3,"v":43},"VEV_4000":{"m":1247.7,"s":18.1,"lo":1189.0,"hi":1302.0,"sp":20.75,"v":21},"VEV_5100":{"m":160.9,"s":16.1,"lo":111.5,"hi":205.5,"sp":4.17,"v":39},"VEV_5200":{"m":89.0,"s":13.3,"lo":51.0,"hi":122.5,"sp":2.76,"v":45},"VEV_5500":{"m":4.7,"s":2.2,"lo":0.5,"hi":10.0,"sp":1.11,"v":44},"VEV_4500":{"m":747.7,"s":18.1,"lo":690.5,"hi":800.5,"sp":15.79,"v":17}},"smile":[{"day":1,"S":5247.5,"4000":1247.5,"4500":747.0,"5000":253.0,"5100":166.5,"5200":96.5,"5300":47.0,"5400":17.0,"5500":7.5,"6000":0.5,"6500":0.5},{"day":2,"S":5260.5,"4000":1260.5,"4500":760.0,"5000":264.0,"5100":173.0,"5200":99.5,"5300":49.0,"5400":15.5,"5500":6.5,"6000":0.5,"6500":0.5},{"day":3,"S":5253.0,"4000":1252.5,"4500":753.0,"5000":255.0,"5100":164.0,"5200":89.5,"5300":40.0,"5400":12.0,"5500":3.5,"6000":0.5,"6500":0.5}]};

const COLORS = { d1: '#06b6d4', d2: '#a78bfa', d3: '#fb923c', accent: '#f43f5e', green: '#22c55e', muted: '#64748b' };
const DAY_COLORS = { 1: COLORS.d1, 2: COLORS.d2, 3: COLORS.d3 };

const Tab = ({ active, onClick, children }) => (
  <button onClick={onClick} style={{
    padding: '8px 20px', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600, fontSize: 13,
    fontFamily: "'JetBrains Mono', monospace",
    background: active ? '#e2e8f0' : 'transparent', color: active ? '#0f172a' : '#94a3b8',
    transition: 'all 0.2s'
  }}>{children}</button>
);

const Card = ({ title, children, hint }) => (
  <div style={{ background: '#fff', borderRadius: 12, padding: '20px 24px', border: '1px solid #e2e8f0', marginBottom: 16 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
      <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: '#0f172a', fontFamily: "'JetBrains Mono', monospace" }}>{title}</h3>
      {hint && <span style={{ fontSize: 11, color: '#94a3b8', fontFamily: "'JetBrains Mono', monospace" }}>{hint}</span>}
    </div>
    {children}
  </div>
);

const StatBox = ({ label, value, sub }) => (
  <div style={{ background: '#f8fafc', borderRadius: 8, padding: '12px 16px', flex: 1, minWidth: 120 }}>
    <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>{label}</div>
    <div style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 11, color: '#64748b', fontFamily: "'JetBrains Mono', monospace", marginTop: 2 }}>{sub}</div>}
  </div>
);

const StrategyCard = ({ title, desc, profit, risk, items }) => (
  <div style={{ background: '#f8fafc', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0', marginBottom: 12 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
      <span style={{ fontSize: 14, fontWeight: 700, color: '#0f172a', fontFamily: "'JetBrains Mono', monospace" }}>{title}</span>
      <div style={{ display: 'flex', gap: 8 }}>
        <span style={{ fontSize: 11, color: COLORS.green, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{profit}</span>
        <span style={{ fontSize: 11, color: COLORS.accent, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{risk}</span>
      </div>
    </div>
    <p style={{ fontSize: 12, color: '#475569', margin: '0 0 8px', lineHeight: 1.5, fontFamily: "'JetBrains Mono', monospace" }}>{desc}</p>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      {items.map((item, i) => (
        <span key={i} style={{ fontSize: 10, background: '#e2e8f0', borderRadius: 4, padding: '3px 8px', color: '#334155', fontFamily: "'JetBrains Mono', monospace" }}>{item}</span>
      ))}
    </div>
  </div>
);

export default function Dashboard() {
  const [tab, setTab] = useState('overview');
  const [selectedDay, setSelectedDay] = useState(0); // 0 = all

  const priceData = useMemo(() => {
    const vfe = DATA.VELVETFRUIT_EXTRACT.filter(d => selectedDay === 0 || d[1] === selectedDay)
      .map(d => ({ t: d[0] + (d[1]-1)*1000000, mid: d[2], spread: d[3], day: d[1] }));
    return vfe;
  }, [selectedDay]);

  const hpData = useMemo(() => {
    return DATA.HYDROGEL_PACK.filter(d => selectedDay === 0 || d[1] === selectedDay)
      .map(d => ({ t: d[0] + (d[1]-1)*1000000, mid: d[2], spread: d[3], day: d[1] }));
  }, [selectedDay]);

  const acfData = useMemo(() => {
    return DATA.vfe_acf.map((v, i) => ({ lag: i+1, vfe: v, hp: DATA.hp_acf[i] }));
  }, []);

  const smileData = useMemo(() => {
    const strikes = [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500];
    return strikes.map(k => {
      const row = { strike: k };
      DATA.smile.forEach(s => {
        const intrinsic = Math.max(0, s.S - k);
        const tv = s[String(k)] - intrinsic;
        row[`d${s.day}_price`] = s[String(k)];
        row[`d${s.day}_tv`] = Math.max(0, tv);
        row[`d${s.day}_iv`] = intrinsic;
      });
      return row;
    });
  }, []);

  const stats = DATA.stats;

  return (
    <div style={{ background: '#f1f5f9', minHeight: '100vh', padding: '24px 16px', fontFamily: "'JetBrains Mono', monospace" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>IMC PROSPERITY 4 â ROUND 4</h1>
          <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>Quantitative Analysis Dashboard Â· 10% Sample Data Â· 3 Days Ã 12 Products</p>
        </div>

        <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap', background: '#fff', borderRadius: 8, padding: 4, border: '1px solid #e2e8f0' }}>
          {[['overview','Overview'],['underlying','Underlyings'],['options','Options Chain'],['acf','Mean Reversion'],['strategy','Strategies']].map(([k,v]) => (
            <Tab key={k} active={tab===k} onClick={() => setTab(k)}>{v}</Tab>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          {[0,1,2,3].map(d => (
            <button key={d} onClick={() => setSelectedDay(d)} style={{
              padding: '6px 14px', border: '1px solid', borderColor: selectedDay === d ? '#0f172a' : '#e2e8f0',
              borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
              background: selectedDay === d ? '#0f172a' : '#fff', color: selectedDay === d ? '#fff' : '#64748b'
            }}>{d === 0 ? 'All Days' : `Day ${d}`}</button>
          ))}
        </div>

        {tab === 'overview' && (
          <>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
              <StatBox label="VFE Mean" value={stats.VELVETFRUIT_EXTRACT.m} sub={`Ï=${stats.VELVETFRUIT_EXTRACT.s} | spread=${stats.VELVETFRUIT_EXTRACT.sp}`} />
              <StatBox label="HYDROGEL Mean" value={stats.HYDROGEL_PACK.m} sub={`Ï=${stats.HYDROGEL_PACK.s} | spread=${stats.HYDROGEL_PACK.sp}`} />
              <StatBox label="ATM Option" value={`VEV_5200`} sub={`mean=${stats.VEV_5200.m} | spread=${stats.VEV_5200.sp}`} />
              <StatBox label="Lag-1 ACF" value="-0.16" sub="Mean-reverting signal" />
            </div>
            <Card title="Product Universe" hint="12 products">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['Product','Mean','Std','Low','High','Spread','Volume'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(stats).filter(([k]) => !['VEV_6000','VEV_6500'].includes(k)).map(([k, v]) => (
                      <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#0f172a' }}>{k}</td>
                        <td style={{ padding: '6px', color: '#475569' }}>{v.m}</td>
                        <td style={{ padding: '6px', color: '#475569' }}>{v.s}</td>
                        <td style={{ padding: '6px', color: '#475569' }}>{v.lo}</td>
                        <td style={{ padding: '6px', color: '#475569' }}>{v.hi}</td>
                        <td style={{ padding: '6px', color: v.sp < 3 ? COLORS.green : v.sp < 10 ? COLORS.d3 : COLORS.accent, fontWeight: 600 }}>{v.sp}</td>
                        <td style={{ padding: '6px', color: '#475569' }}>{v.v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}

        {tab === 'underlying' && (
          <>
            <Card title="VELVETFRUIT_EXTRACT â Mid Price" hint="Underlying asset">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="t" tick={{ fontSize: 10 }} tickFormatter={v => `${Math.floor(v/1000)}k`} />
                  <YAxis domain={['dataMin-10', 'dataMax+10']} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={v => v.toFixed(1)} labelFormatter={v => `t=${v}`} />
                  <ReferenceLine y={5247.6} stroke="#94a3b8" strokeDasharray="5 5" label={{ value: 'Î¼=5247.6', position: 'right', fontSize: 10 }} />
                  <Line dataKey="mid" stroke={COLORS.d1} dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
            <Card title="HYDROGEL_PACK â Mid Price" hint="Mean-reverting, wide spread">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={hpData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="t" tick={{ fontSize: 10 }} tickFormatter={v => `${Math.floor(v/1000)}k`} />
                  <YAxis domain={['dataMin-10', 'dataMax+10']} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={v => v.toFixed(1)} />
                  <ReferenceLine y={9994.7} stroke="#94a3b8" strokeDasharray="5 5" label={{ value: 'Î¼=9995', position: 'right', fontSize: 10 }} />
                  <Line dataKey="mid" stroke={COLORS.d2} dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </>
        )}

        {tab === 'options' && (
          <>
            <Card title="Options Smile â Time Value by Strike" hint="Snapshot at t=50k each day">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={smileData.filter(d => d.strike >= 5000 && d.strike <= 5500)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="strike" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="d1_tv" name="Day 1 TV" fill={COLORS.d1} />
                  <Bar dataKey="d2_tv" name="Day 2 TV" fill={COLORS.d2} />
                  <Bar dataKey="d3_tv" name="Day 3 TV" fill={COLORS.d3} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Options Pricing â Full Chain" hint="Intrinsic + Time Value decomposition">
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={smileData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="strike" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="d1_iv" name="Intrinsic (D1)" fill={COLORS.d1} stackId="a" />
                  <Bar dataKey="d1_tv" name="Time Value (D1)" fill={COLORS.accent} stackId="a" />
                  <Line dataKey="d1_price" name="Total Price" stroke="#0f172a" dot={{ r: 3 }} strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Key Observation: Time Value Decay">
              <p style={{ fontSize: 12, color: '#475569', lineHeight: 1.7, margin: 0 }}>
                VEV options are <b>European-style call options</b> on VELVETFRUIT_EXTRACT. Time value peaks at ATM strikes (5200-5300)
                and decays across days (Day 1 â Day 3). Deep ITM options (4000, 4500) have near-zero time value â they track the
                underlying 1:1. OTM options (5400+) are pure time value with declining premiums. The VEV_5300 strike is at-the-money
                since VFE oscillates around 5248. Time value across all strikes shrinks from Day 1 to Day 3, suggesting approaching expiry.
              </p>
            </Card>
          </>
        )}

        {tab === 'acf' && (
          <>
            <Card title="Autocorrelation of Returns" hint="Negative lag-1 = mean reversion">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={acfData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="lag" tick={{ fontSize: 10 }} label={{ value: 'Lag', position: 'insideBottom', offset: -5, fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 10 }} domain={[-0.2, 0.05]} />
                  <Tooltip />
                  <Legend />
                  <ReferenceLine y={0} stroke="#94a3b8" />
                  <Bar dataKey="vfe" name="VFE" fill={COLORS.d1} />
                  <Bar dataKey="hp" name="HYDROGEL" fill={COLORS.d2} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Mean Reversion Signal Interpretation">
              <p style={{ fontSize: 12, color: '#475569', lineHeight: 1.7, margin: 0 }}>
                Both VFE and HYDROGEL_PACK show strong <b>negative lag-1 autocorrelation</b> (-0.16 and -0.12 respectively).
                This means when price ticks up, it is statistically more likely to tick back down at the next step, and vice versa.
                This is a textbook mean-reversion signal. Beyond lag 1, autocorrelations are negligible (noise), confirming
                that the mean-reversion effect is a short-term microstructure phenomenon. A market-making strategy with inventory
                management would exploit this â quote around fair value and let the natural bounce-back generate P&L.
              </p>
            </Card>
          </>
        )}

        {tab === 'strategy' && (
          <>
            <Card title="Recommended Strategies â Ranked by Expected Profit">
              <StrategyCard
                title="1. Market Making on VFE & HYDROGEL"
                profit="HIGH PROFIT"
                risk="MED RISK"
                desc="Both underlyings mean-revert (ACF lag-1 â -0.15). Quote bid/ask around a rolling EMA fair value. VFE has tight 5-tick spreads and 75 avg volume â tight market making works. HYDROGEL has wider 16-tick spreads â more profit per fill but slower."
                items={['EMA fair value','Inventory limits Â±20','Spread quoting','Position fade at limits','Skew quotes by inventory']}
              />
              <StrategyCard
                title="2. Options Delta Hedging / Arbitrage"
                profit="HIGH PROFIT"
                risk="LOW RISK"
                desc="VEV options track intrinsic value closely but time value varies. Buy underpriced options (when TV compresses below average) and sell the underlying to delta-hedge. Deep ITM options (VEV_4000, VEV_4500) have very wide spreads (21/16) but move 1:1 with VFE â synthetic long/short the underlying."
                items={['Delta hedge VEV_5200','Buy VEV + sell VFE','Monitor time value','Cross-strike arb','Put-call parity checks']}
              />
              <StrategyCard
                title="3. OTM Options Market Making"
                profit="MED PROFIT"
                risk="LOW RISK"
                desc="VEV_5300-5500 have tiny spreads (1-2 ticks) and 40+ avg volume. These are pure time value with no intrinsic. Market-make around a Black-Scholes fair value. The time value decay from Day 1â3 means net short theta is profitable."
                items={['Sell OTM calls','VEV_5300 spread=2','VEV_5400 spread=1.3','Theta decay edge','Small position sizes']}
              />
              <StrategyCard
                title="4. Cross-Day Momentum on VFE"
                profit="MED PROFIT"
                risk="HIGH RISK"
                desc="VFE opened at 5245 (Day 1), drifted to 5267 (Day 2 open), then reversed to 5232 (Day 3 close). Day 3 shows a pronounced downtrend in the middle section. Use a longer-term trend indicator to take directional bets."
                items={['20-period EMA crossover','Trend following','Higher risk, higher reward','Requires timing']}
              />
            </Card>

            <Card title="Backtester Guide â prosperity4btest">
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
                <p style={{ margin: '0 0 12px', fontWeight: 700, color: '#0f172a' }}>Setup:</p>
                <code style={{ display: 'block', background: '#1e293b', color: '#e2e8f0', padding: 12, borderRadius: 6, fontSize: 11, marginBottom: 12, whiteSpace: 'pre-wrap' }}>
{`pip install -U prosperity4btest

# Run on round 4, all days:
prosperity4btest my_algo.py 4

# Run on round 4, day 1 only:
prosperity4btest my_algo.py 4-0

# With visualization:
prosperity4btest my_algo.py 4 --vis

# With custom data (your 10% sample):
prosperity4btest my_algo.py 4 --data ./my_data/

# Set position limits:
prosperity4btest my_algo.py 4 --limit VELVETFRUIT_EXTRACT:80 --limit HYDROGEL_PACK:80`}
                </code>
                <p style={{ margin: '0 0 8px', fontWeight: 700, color: '#0f172a' }}>Algorithm Structure:</p>
                <p style={{ margin: '0 0 8px' }}>
                  Your algo must define a <code>Trader</code> class with a <code>run(self, state: TradingState)</code> method
                  that returns <code>(result, conversions, trader_data)</code>. The <code>result</code> is a dict mapping product
                  symbols to lists of <code>Order</code> objects. Use the <code>Logger</code> class from the sample to get
                  visualization-compatible output.
                </p>
                <p style={{ margin: '0 0 8px', fontWeight: 700, color: '#0f172a' }}>Order Matching:</p>
                <p style={{ margin: 0 }}>
                  Orders match against order book depths first, then market trades. Use <code>--match-trades worse</code> for
                  conservative backtesting (only fills at prices worse than your quote). Position limits are enforced â if your
                  total orders would exceed the limit, all orders for that product get cancelled.
                </p>
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
