import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class RALO_Datasets(Dataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 subset = "train",
                 transform=None,
    ):
        super(RALO_Datasets, self).__init__()

        self.transform = transform
        self.subset = subset
        csv = pd.read_csv(csvpath)

        if subset=="test":

            self.olabels = np.asarray(csv['TOTAL Opacity'],dtype = np.float32)
            self.glabels = np.asarray(csv['TOTAL Geo'],dtype = np.float32)


        
        if subset=="train":

            self.olabels = np.asarray(csv['TOTAL Opacity'],dtype = np.float32)
            self.glabels = np.asarray(csv['TOTAL Geo'],dtype = np.float32)

            
            
            
        self.images = [os.path.join(imgpath, str(idx) + ".jpg") for idx in range(0, len(self.olabels))]
        test_rep=[1186,1200, 1201,1208,1209,1211,1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1247, 1248, 1249, 1250, 1251, 1252, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1270, 1297, 1298, 1299, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1359, 1360, 1366, 1367, 1443, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452,1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1498, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1590, 1591, 1592, 1593, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1657, 1660, 1661, 1662, 1663, 1664, 1703, 1716, 1717, 1718, 1719, 1720, 1750, 1751, 1752, 1758, 1792, 1799, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1854, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1967, 1968, 1969, 1970, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1996, 2055, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2082, 2083, 2084, 2085, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2116, 2118, 2162, 2163, 2164, 2165, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2196, 2197, 2257, 2269, 2270, 2271, 2272, 2273, 2293, 2294, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2365, 2407, 2408, 2409, 2410, 2411, 2412, 2424, 2425, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2502, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2693, 2694, 2695, 2713, 2715, 2716, 2717, 2718, 2747, 2748, 2773, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2813, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2868, 2869, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2901, 2931, 2935, 2948, 2949, 2958, 2986, 2987, 2988, 2989, 2999, 3000, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3055, 3056, 3057, 3058, 3062, 3063, 3064, 3065, 3066, 3067, 3096, 3122, 3123, 3124, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3147, 3164, 3165, 3166, 3195, 3203, 3204, 3205, 3206, 3207, 3208, 3255, 3256, 3257, 3258, 3259, 3260, 3294, 3295, 3341, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3368, 3369, 3375, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3399, 3400, 3401, 3402, 3403, 3404, 3411, 3441, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3463, 3464, 3465, 3471, 3472, 3473, 3474, 3475, 3477, 3526, 3528, 3531, 3540, 3541, 3542, 3544]
        
        if subset == "train":
            self.glabels = [self.glabels[idx] for idx in range(0, len(self.glabels)) if idx not in test_rep]
            self.olabels = [self.olabels[idx] for idx in range(0, len(self.olabels)) if idx not in test_rep]
            self.images = [self.images[idx] for idx in range(0, len(self.images)) if idx not in test_rep]

        elif subset == "test":
            self.glabels = [self.glabels[idx] for idx in test_rep]
            self.olabels = [self.olabels[idx] for idx in test_rep]
            self.images = [self.images[idx] for idx in test_rep]

    
    def __str__(self):
        return "RALO_Dataset({}): {} images".format(self.subset, len(self))

    def __len__(self):
        return len(self.glabels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])#.convert('RGB')
        glabel = self.glabels[idx]    
        olabel = self.olabels[idx]    
        if self.transform:
            image = self.transform(image)
            

        return image, glabel, olabel

if __name__ == "__main__":
    dataset = RALO_Dataset(imgpath="Lung_Rep1/", csvpath="lung_rep1.csv")
    print(dataset[0])
    print(len(dataset))
