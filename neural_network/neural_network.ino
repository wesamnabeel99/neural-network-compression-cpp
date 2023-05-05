#include <math.h>
#include <stdlib.h>
#include <LiquidCrystal.h>

#define DATA_LED 13
#define SUCCESS_LED 12

const uint8_t IMAGE_SIZE = 28; 
const uint8_t N_OUTPUT = 10;   
const uint8_t KERNEL_SIZE = 3;
const uint8_t STRIDE = 2;
const uint8_t CONVOLVED_IMAGE_SIZE = IMAGE_SIZE-KERNEL_SIZE+1;
const uint8_t POOL_SIZE = (CONVOLVED_IMAGE_SIZE-1)/STRIDE + 1; 
const uint8_t N_INPUT = POOL_SIZE * POOL_SIZE; 
const float WEIGHT_PRECISION = 10000.0;
const float PIXEL_PRECISION = 1000.0;

const int LCD_COLUMNS = 20;
const int LCD_ROWS = 4;

int count = 0;
#define PIN_E 11
#define PIN_RS 12
#define PIN_D4 2
#define PIN_D5 3
#define PIN_D6 4
#define PIN_D7 5
LiquidCrystal lcd(PIN_RS, PIN_E, PIN_D4, PIN_D5, PIN_D6, PIN_D7);


byte input_image_square[IMAGE_SIZE][IMAGE_SIZE];


int convolved_image[CONVOLVED_IMAGE_SIZE][CONVOLVED_IMAGE_SIZE];
float output[N_OUTPUT] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
float kernel[KERNEL_SIZE][KERNEL_SIZE]= {{0.54876,0.42717,0.23532},{0.605,0.321,0.73093}, {0.212,0.97456,0.10124}}; //random kernel
int pool_output[POOL_SIZE][POOL_SIZE];


// trained output_weights
int output_weights[N_OUTPUT][POOL_SIZE*POOL_SIZE] = {{307.1791709292191,-769.9861137511886,889.1925744880921,-276.3559241380101,-425.5112396246646,-2583.9693208855465,-1599.7258887145972,-1172.9971655202044,-2941.2533993268316,-2769.4491289327534,-3115.3469870848785,238.49619760590957,-215.73747468168423,-635.4893858642984,416.9344246833339,-1474.4695581769295,-768.5986445495876,-2954.564842241501,-1290.573546694779,1039.5931097003302,-405.7131271582249,-258.5969084385367,-312.4949770774087,-972.313068874855,798.8562287592381,-808.6280967695501,655.6489411607646,15.278797007946432,-545.6532906501958,-1794.105820544646,784.8159370015936,-930.4969857607257,-314.52826321601896,959.2360501455223,1556.3362507181566,1713.9093587551117,891.8332156383123,-300.408718640566,-1666.67313131072,-744.8693197592623,-775.1911560347811,-2227.25112943007,-999.5092489424048,-1700.6431963841062,-137.71267716321532,1188.2682535378563,1453.156334777671,424.17893382010874,-769.5052020477718,-283.21293842346165,-778.23337863878,-3308.7022917970444,292.03364771530744,-1471.100157648696,-486.21183307993965,-1291.3996169430209,-883.2485442891453,219.42973053456737,-1259.35565058938,-326.0877257202444,114.03078033447788,2883.3852935664804,-1343.0859307460898,2805.0702617554944,-865.6773218530077,-418.7558244318235,-1346.3037272172153,2313.602226988425,1658.336998138088,932.6105190926518,-508.72133792029905,664.8064450402738,-3517.9655989347198,737.2312415826509,1032.0206739438083,295.76153520408667,2126.796993945426,-1188.9957911231188,681.2333394732027,-972.1997795762334,312.3241895147704,1309.9982952527769,-1743.0649056709763,2694.3785369000075,-2757.6152022584774,-5164.451550980371,-712.8151784404805,-1561.459243530358,-9.262209875353614,1208.7180200695557,8.031795848456628,-411.0814464287817,-588.4601782515022,1658.3422124637143,1459.1842116085552,202.38661436342545,969.0268969850066,-6626.442080320981,-4498.926099683281,1626.1054857101878,26.596378531235114,-293.4628091423148,692.2008874533759,1603.9154746523502,132.35526553569966,-1915.1399246526116,328.5222470243995,815.2825993549086,2607.596040346017,2012.3518956024554,-5284.386524152639,-1231.6705022958247,939.5994680404551,-1225.9127993475252,-343.2978560125111,1055.7161253870124,-560.0022400160133,196.43519553898065,-2604.0488051812304,-977.7966596596103,1061.2182941464305,-177.5428163966636,1974.3934656793058,-918.9130152333033,-1746.0683888474475,-1678.3730804965878,-314.6733303300566,-1943.9933071162848,168.2856630302838,640.9104731366276,475.73099815367726,-1265.7910928716165,703.235595619224,565.9020767381338,-1186.2941743589847,1669.769052005731,243.28722761142376,-805.3688057090934,-976.5160529347005,-1988.7912204340078,-337.54585814450604,-915.6915669253409,28.71621187579561,-104.34394832954358,147.93505496257663,534.7889936167545,349.5835183481506,-21.585357002836762,2331.6256275841056,591.8767733112538,414.64931481731224,-3008.700996014009,-873.4824794747865,217.32089850381342,83.8650448287517,390.0666977615321,365.26738855479135,439.7200043303291,-406.06803935278367,-2493.119975710125,-2944.5208227456615,-3373.313772729498,-2826.2712775494306,-2293.6247167528895,-2591.908744376058,-1190.5774026209444,-1146.498613337622,640.9541396194645,-114.78331946934819},{-601.0178756137103,252.0637435703692,552.7598437853608,-150.19502440558657,-1523.1473923392261,-665.3501477402363,-1060.744169709184,-1608.7472601298,-2661.1740937910567,-962.8657195583633,-825.3841876737014,-744.8196650332227,-519.1245736037112,-416.37845507912823,-299.6630428805419,1265.502155122049,308.3633883551641,-935.5582136246728,-2490.778003234992,-355.75184609054617,183.96719834122013,-700.7350513731693,-1278.0122785214771,-2141.5846698225055,1952.2366180594352,-129.92838474897474,787.0128002937045,-498.85892411229855,-695.3638611699872,1383.8640232203536,-809.8342145864849,-436.433623259639,-1336.461540055078,-1444.76123934753,404.78605658253065,1142.432207110459,1523.8408123081329,-1998.373358288214,-631.0294079359651,494.78720793540344,-237.8967192890768,-2402.8662604572737,-3784.096810334393,-985.4246434147104,-2387.476776933304,-3001.6371621406565,-887.3124187411818,-3403.6830544427976,-37.50110944763282,-2050.8876067313167,-1938.3424573431466,-2135.747119499286,201.71382584600053,-929.369089229831,-2059.5392061804887,-937.2447371964846,-406.1144417055031,-81.51911826939258,3093.2422411863663,1434.5917138047323,1512.5659145430707,13.491260793217881,-3164.3345644529513,-1858.1213647368813,614.7915524345373,674.1884132906636,-458.0809896246643,1936.7754925565678,291.57675634792093,-1867.2105559953086,-2595.031709170347,1623.1220174310736,2479.2830173156794,1650.6397080620493,-3023.927373832384,-2515.6506476776335,83.86314225482398,-358.1318649418804,-587.3890122949372,820.371234733422,-138.66474960226623,11.155320438255066,-3365.610025370513,-523.9710261719264,2200.227360626708,1911.3388469627585,-219.68974867514655,-1470.1052828472107,-1149.074139162051,663.5611208756965,404.5449913668924,-312.8860072575107,-24.321740952573222,424.62880467911793,1413.1745034565388,-2565.242023527305,-1260.1022935341575,3288.378360030648,-1586.3914244977602,-491.728889597487,-3640.66478585708,-1064.96358781342,287.61485420887124,270.2914632925083,-454.29601923911264,603.9715047324372,-2020.0072539251237,-2842.650181414932,-1440.3979116262135,-1193.7451474647926,2320.36451713883,-997.6806317944262,-1654.9045963476688,1204.2141233959965,-520.8860907336455,-511.77202292280856,391.23254915124363,-982.8850512291104,-1541.315084351999,-1042.787996513877,-852.6249040191116,-1052.16918375103,431.56461285832717,120.92114228730223,1888.2108227995607,-1917.4772845542236,1008.2197755098651,1344.4247525529584,-186.50355331277893,973.8300443081353,949.4169225581469,-636.010191916379,1459.0469624646935,2026.434275614517,-0.2012889956012613,-1049.6565828704245,-1064.997659677011,-1509.908768846647,1341.1863741562343,-411.9555020080515,-1804.2860830268987,-1205.3969392623535,690.9144980736424,973.0865717650704,-736.6738014650241,856.3911975025676,213.91331795271614,-361.2454970584572,-620.0542471136996,-75.3281333478229,293.7114046976798,2347.1982428108845,534.6383583735833,-500.32691345635004,-261.5082878552644,767.1925150739595,-350.0080245608372,587.9471587050141,-1315.7476835207437,-1777.0025334726158,-1590.2036565020628,-4271.726825119335,-2182.881376757661,-2623.1864290166263,-1275.0169440872792,145.9279221349539,-788.382067888208,-148.386649614124,-605.627771814517},{-816.5871861056814,-834.1549787563705,-970.8141937621352,531.6857171005269,3206.807310579952,-1329.2856204599784,-1238.1508331766304,-4372.717248891527,-3199.0103489869525,-1691.6452474107816,-49.744190477092125,-681.6980204406199,-276.2881963025354,561.6382281194124,815.7189076576764,772.1046938600653,2896.204980948841,3.3234716751017084,-403.07998348448905,2107.9351302362666,240.7873651943677,379.0794467583843,-671.4124452799144,-1779.5621489400655,-714.2433181774803,937.2540690416245,99.2784159787457,1011.2844176660569,-1506.1600369387995,-310.57060538075376,-1078.063489343274,-1041.4971019603663,1695.4504191799178,631.7820665973079,7.245669809948311,-1433.9709162249933,1250.7948921646712,-2343.829149305927,-1402.7246060415646,-957.0109862510513,650.0676974959929,-1619.6435246786716,1042.1308565831876,962.3164135890476,1596.6966371488843,563.7070610215841,-2424.7484301695267,-1299.3944699870951,1596.179873003761,-593.1990409156551,-27.105963659193538,-2090.75855373865,-1152.8920357753934,1712.4698533066303,1705.4080645010085,953.1747549806755,-2947.765697327167,-339.6538514118598,-3344.731773127637,2048.423086738992,-1176.7994219041313,-85.21864899092253,-335.5743713978943,-1983.3002714754418,-2459.5185773220646,-1105.5208326660504,-2429.4913531032234,-2941.466629813428,-5681.479027556125,-4797.599513875835,-1356.73429041734,-4836.889264040068,-3056.6225376873317,-576.3566597221834,-159.68142898368558,2583.6872000249996,-1272.234147673044,463.85703820059194,100.58204983351938,-115.84885775466601,-2973.5976246200175,1162.638350647268,94.61578731087904,-182.1715257482707,-249.30705249603065,-657.1805772299595,866.4453484826146,-2875.689637593665,-1796.7547056225435,-4338.843078137722,1945.8527844796981,904.5128690341265,1593.3422720991402,546.6014841750449,-314.2671058566855,762.225396252202,333.68833087707264,1771.4596450916176,-905.7351932169005,230.8971372720637,502.092429337249,-103.58567748604023,2014.5348996344405,2381.445524781858,-71.44929247761762,-652.5925436245487,1349.9723425288971,137.85043810842384,-87.80420305194413,323.90679886254117,2741.081731600037,-1556.827214332502,1223.5516978029686,-2472.946887428376,1589.2090708751336,960.9993525308175,1178.6961861897005,-1264.59419563848,-2117.3938562970147,1041.132817371579,1189.114641446304,1799.7048242339329,1047.5254292067043,477.5149460530877,1013.6469039032181,3454.8183433123163,-1445.7942478879856,2834.0669189541145,1701.8326389646465,-877.2584932001972,724.6140622925633,354.39148664955883,2123.086469146322,-1379.7447107695434,497.64889546885394,2598.0537040615463,-2095.9761167494125,-3049.4436066705307,-1732.4222322885284,4.857056956266885,1827.0582291682592,1348.3833202401033,-578.4627922595105,623.9352081077018,890.873640587847,-1168.9684186583218,663.925150928187,-2665.181588668187,-1164.811677322509,778.5912539954953,-1292.5254444151797,1284.9011427389485,1933.3748285186682,663.4867292169487,164.4918597241035,-957.7842010809779,307.88864925207537,-768.7695153647414,-635.2139654901605,-2189.483429045154,-1832.9318018121232,-3402.869180394884,-1904.2350705879462,301.0939686783593,723.9757879892953,611.4685956383684,-1115.352796231892,-299.06156510717256,677.5215238697149},{454.3809081845176,62.50921358908457,872.2652101739252,-594.9560780256576,-192.21288307917806,1787.5850955835276,747.3788907862511,1987.315185519313,2356.3905694657947,1414.1270729462885,616.3712964742459,-304.51288699865,574.1869340222019,-879.6950110953758,-102.38358620722914,-532.4296727603611,-1337.3791381514402,-2303.8479152029086,-1387.9014572848794,-2544.4339892753446,-609.378466367572,-1232.935594644731,-1738.9339219029448,-4434.710983272195,-1310.062945331341,646.3698462606633,641.9742708980259,947.9851112447974,2562.8890944488007,-795.1973077607872,680.7651211165355,2795.9298574607033,2443.565984106764,-152.19791155236257,1670.7384119839417,-170.7334478033958,-2452.1681435963364,-385.48303727258207,-2014.9444938323854,386.0099217679674,1409.403792865726,-507.7174451904514,489.1584024031474,1409.446750635275,-1609.0765818860896,-2633.015974030995,143.72051324684404,-1586.4589117453884,-519.4518930322291,2256.644208764473,-1253.621277719761,-3716.6511453524154,-211.8945560324575,1418.4195360112046,-367.80817025788974,-640.7321857446133,-2114.7042267025636,-1497.0412966570907,1021.2348726075287,-387.02294876591975,3710.716526927901,379.3923601973927,140.62610574453547,-311.12708691008334,-3158.908104591268,-1383.802599983483,-1126.889199579224,-937.4642131160407,-627.2819917746702,-3157.3182733416766,-1500.8554911265483,943.7864964415003,1898.9168679466075,-403.5149256454007,101.22729776576399,-3087.309455109446,-1272.7209351895171,-158.89689996317966,-215.9054281142633,-960.8447488664332,-4095.1841411375467,-461.64155739261554,514.8700396883634,1797.1979680319307,980.9094330022272,-388.17215089100273,-1898.333753040304,405.0807779486084,-26.40598546620746,-2033.3527711966149,102.52569244122081,-402.03276961124624,566.9342460168848,-3700.444350299173,-1760.8157860618169,-594.1263640132206,-106.9584019693921,-1208.909870868008,-804.1411582162243,371.0285973058149,955.9868551341408,344.20168094862385,-483.20575666862806,1011.0826286889278,-934.436772319838,2184.2428526735757,1951.0042444427897,303.83417617051094,-1388.5296358290204,-3608.7627881781427,-3417.3983667780776,349.8923923496558,814.92893305283,536.3572044970535,-298.36574055533555,565.5241567940742,-1678.584868185816,-206.12461203413926,2555.462494531382,3103.8680002883953,-1953.4846693399556,466.45350807884927,-889.4159515380575,-810.3975151852367,-2428.0227999793365,2431.988222966046,-1949.4943774233534,430.25731976085444,-44.70558507768536,-2506.723197618934,848.1815025693903,128.08700629041203,-491.8725623993062,941.8893184827424,912.7465769776003,-2113.6521619551118,971.0586808096614,2046.119319167274,-482.42114931809976,1228.2907357445793,4064.269490628493,-1094.5367049868207,-2129.633947088289,535.2766943157312,3866.6565415476375,2911.459817398017,-414.88767918412793,373.7524047054724,59.96840486747031,863.0480455278009,-1704.6782952900253,-2101.690874457166,-158.40571725851862,-1072.1684327804503,-2395.9172358864794,234.03026134412048,248.17811548587298,256.4881477250139,1982.8370557127648,3641.45222541986,931.0732410014714,-1492.7598759531204,-858.7326314587731,-2289.782962221938,-1467.4873176914919,137.55938468943333,-28.434217176588653,-658.3268896670833,-1171.3278781017486},{873.1054104697862,-146.685833625809,686.1275975834778,-400.79810305036784,-1462.5358519426252,-3862.656187185311,-3782.895367501577,-3054.625098377619,-3867.3597203236413,-756.2286312895797,-753.0422238213201,-1017.5133922593795,607.5765364876755,-64.16541582742192,34.35958273136818,-513.8670330281569,-811.2532939245377,-385.18998794386005,-2948.854356976776,-731.3921099392452,831.4707991694478,125.9432955197591,-1492.031598429117,474.7722786947997,813.9152035102671,-272.3591696572943,543.6099093067455,71.04155687328483,-561.1597798074038,102.2148922070073,177.03614051790663,378.42663943261704,-2086.111714199067,475.99522544433944,-562.8078736290527,-784.7991463281352,1811.2992578845556,2465.5407606681792,-936.7776250658683,-758.07124116839,1670.6856777055416,1028.4004736660834,-804.167466517853,-1340.0916756492088,4.422112166827286,-5060.332655770507,-2238.845896402614,-2686.913161941339,1112.7850474592353,880.3578165219891,52.85268329486271,-1131.4179269086308,-1990.8982355356172,-2320.4281545132408,-2881.1984743517537,713.9709559515426,-2605.6095727012384,96.00404126518407,-1869.6705593594934,924.7190799883803,396.63462840610293,-872.5277796555034,-1998.0337947846238,-309.4169255417334,-2358.9924830917234,-1186.1401920603735,-625.1010519058857,-1045.6056446326063,895.3993564739387,2261.754660260502,3792.7355718743747,-1632.6985174182946,-2408.37795600164,3567.970435470527,-2761.7472010157426,303.1511985242569,1192.9913159782175,-1078.5989736960573,-519.2858894765071,93.70526867245974,3400.17244685766,662.6042594453913,-133.0632411893493,2843.5875354204204,-214.26546522633913,-1636.151957792909,3239.804838983017,1117.6841924389066,517.2566106736467,-847.9383332702741,-1940.6350888093934,-616.1545731825751,-3277.5887501041325,-2413.0772991485974,1868.173762600994,1711.143593675978,-581.512434123125,144.631986991706,1959.7711868159695,1540.1121070090924,-29.208149549496564,-141.05297348305788,1458.8839846478747,-906.858418914833,-1047.6099138486472,-2269.905613072865,1280.5646865955623,1158.6197299530397,5.104298612804813,-817.8537513082322,1185.3443080093152,1661.266668690792,-713.7963940689796,216.38381541729507,-607.7562912866771,-874.1957757308277,1115.0242745582725,471.25970743436915,-475.18878609788095,-650.3843478952967,-2729.000298821394,-1872.000330302328,-2506.608545451338,2896.401812533,-376.71358320048176,-1820.8895313926791,-1101.7211289073,-573.6580859636734,-2153.8175813728744,-2310.1832945453043,788.9168978717041,-343.2696238684519,-1364.8698902762706,-559.5998681254596,-2393.6441952326486,-1488.9251126543302,-3440.423822522305,103.71785081059274,-338.6036113866325,1714.1459266168379,636.4871792343879,1568.0743547971267,-770.8754785660574,707.1494535446761,-943.5751908510538,240.17633719831147,-11.088106764452698,328.433643385317,907.4891529323914,2127.707837654788,-162.6590285825083,1870.300161828782,910.7777236267633,-1876.146794674697,1900.9577066941938,-351.90843198402524,-335.66604305643625,833.9271432555747,-1183.9000390629326,-1643.2286786149737,-2371.455742137945,-4568.732123105086,-4896.949306240962,-2917.859918691571,-4677.620188379228,-2824.0063027206784,-3499.3832380032613,-1210.8746505935676,625.9266079895315},{579.3148057032178,-683.4166857722923,58.462279611149114,-182.3524201679947,-566.7642325761508,-60.055964737627995,-2229.2934650000825,-2714.468250944931,-2990.446236603389,-2831.491011095484,-1031.7185099577323,-187.8757964511935,962.6473873051737,781.4549264198833,-368.709551391354,-1152.974576897059,-1620.5458305006546,-13.630046239351909,-2049.8122217612267,588.7405119434585,-774.5301946721479,236.243040675553,902.8149907445561,2233.1235203548276,-246.55123316126327,1401.8015559526575,214.7074211871185,-720.7986174898899,-4866.8771465508935,-947.7496726585722,-329.9521277238891,3335.183347378746,-2701.8797402112823,-1831.139272070892,-1590.2518806482237,1139.5584118968734,-1474.2056902996317,209.36650514522563,-203.20518310051563,-518.9652973756206,-3115.2397933304146,-2949.80827239435,52.91571688067988,-178.58974713970042,-2619.349135206865,1027.1851298371585,1075.1699611193446,-329.51090995103726,280.1552262961817,2078.827657182152,1308.9337337397315,5583.345532458173,-450.25653768326634,-2444.041078505273,155.23586173477696,1182.911256218684,1409.1705048418157,2949.708621592257,141.40483176691177,-1911.8012394842895,-3479.8060221430796,-1359.6767529013187,-1782.3727983022202,385.0667692068824,7439.342960797535,-285.04450717245135,-967.6372257501015,2231.6670020946394,1486.145049227514,-2137.6306019234394,1099.9485339837645,1939.6653196172444,-3016.827659627258,638.2522243561175,-1689.2278974802155,-4715.874580421497,-6367.943809116525,-763.3516268228366,798.2134698037484,480.3324315322824,-1666.5795927788681,-960.7935935989585,2447.3260043535265,462.28942308673396,-568.1175817888297,207.98756647373156,-767.1191540106898,-1647.511639733074,515.1530118520136,-5277.16253317136,-2232.6463417121986,47.50229396786232,-5.4837135507350485,-507.5534231252662,-1046.3570192959592,-1849.5077224715155,1864.7171256865179,-1935.405063884112,-1320.5153180823595,-2088.5151030839015,585.5922979788694,1698.8471919411027,-1585.0993998967554,-75.5351007476922,728.6559648909205,-281.5717026727362,975.4280194614188,-1911.4216215064473,-2298.458205053248,-818.3476649587312,-2864.459623550222,1741.6405518573908,677.8888728575172,254.2462374869376,-1432.8883869103424,281.89353543539676,-545.836771179983,1721.4651330988254,573.3742367224016,-471.41002178983473,7492.9260538860235,-1221.7935781005276,1030.3919243053074,-2.747661099023601,-214.28097477777786,92.33923852493676,852.2334020077249,1391.9364381048665,2232.5572690135973,-1408.083284808817,-817.7161086351606,1444.8262353258126,-2197.451435237665,-2184.363805365096,-574.4925555842594,1522.8572867999862,-592.8596631367687,-1566.4480713614441,171.24298436686954,692.4334237522216,1565.3531177431314,215.25989771421087,1167.5598303433835,-579.7168868118746,-550.2578812762699,979.3606387549544,1374.7938715160751,478.896346467774,1457.2008183219923,1305.4561532803968,2223.990034936487,154.20176984355533,-542.1413696266161,-456.93242607240035,1964.5585624331954,-666.4540201329303,142.65264426310333,229.11398187546675,-1108.4608037728547,-1617.0526294636857,110.74455957731848,435.296605206021,-983.5915470613021,66.17466980259687,-2838.4622578937624,-381.4472522256726,-1379.241813386669,-1704.1970814060192,-994.5298453269452},{-293.9967211808783,631.0245638200797,-39.021222318933795,-264.9301655186645,607.1325601382287,1378.385944083591,798.1516923943379,716.2959346850544,-248.18321499818117,2409.324624097487,-189.13290225950286,-276.8224229568371,-974.2376098471675,138.45142662999808,1155.6334657667817,2584.577973717327,-876.8343674606404,205.75965624596907,1604.0646545109364,-3668.9972001480373,-1727.2910691670945,1134.854113183385,2048.229878879649,-107.86970313566322,-1869.4627182854417,-825.4285850727196,412.9182365978017,-897.6309261054902,-832.5232118695146,-2380.877970802628,-254.12695206191748,-2832.7828149542643,657.8010715942647,518.3258069605799,-1632.6196587407762,-927.1240329602914,1908.6320888146881,-259.01495357382527,419.1172733201632,454.0799782233632,-1990.3370181873445,-1922.099076058333,209.94505228985633,-1370.621439869165,-2528.394014492235,395.5452449171035,-1100.9625236918043,-2028.05556775354,-2057.87413802429,-3727.7934588097864,-2373.184093995929,-1002.3812513025402,49.716506232090396,-517.4516442336034,-829.8562879951414,507.28799410635696,-1004.0856049346611,224.2161075533272,-427.9055349118358,-1528.1257988819186,-2531.6613116942262,-4164.91275051577,54.648282905695936,-1115.9288003861557,-1788.6390075354732,122.92362114185939,264.8341259533353,193.8687868465934,777.2959109231473,725.0226414230103,2153.47371403418,-1604.2057990207265,-1216.0193263829124,-2208.915509139649,2092.2685541009987,229.56215604649861,370.7059046891379,2381.89885599714,-1028.8965519213493,1305.409296825787,816.6649830378364,1514.3496964138537,363.65495582615983,-2068.789278126383,2721.4429515814186,1144.0119257161418,-232.28837461952043,-2896.721087302201,2699.2194292532286,1572.5847620862655,641.6674344142089,893.0573252449062,91.72448110716316,-1452.8140357641644,366.21143910001877,1336.747381094434,1418.40347324204,-535.8103298615682,-1523.475598631081,-344.042158541425,1095.5341177677885,875.4710298061386,1082.6639684515276,-3413.336408744609,117.64874145218255,-1530.772872792529,-3209.5964838418695,-401.77238916398306,3049.029938626697,771.107590562359,1224.8570744542346,-2230.894597542423,-84.67743464271439,-721.613010158208,-1982.4478606958253,153.34529477990492,-2452.914334602661,725.6568007533747,313.471287080651,-1936.4194023141781,-1940.0826045626704,1229.3671261843178,4134.943861105444,-1290.615979421889,2094.040324580677,866.4992736648503,934.9089040759499,-1085.4225844425068,-1586.5963152546335,-402.60780172700913,479.3429304734825,-951.1238478011874,-805.421140734511,1015.4272901690449,1356.1822644594733,-1569.9885742080535,2242.460213953215,1365.7554717237508,1359.0726541630947,472.29168347232303,-910.8763878184146,-84.95671415505349,-37.13746452733807,299.2743401414903,-784.0461508742357,-658.609742090833,-2732.6691293065714,-2318.4621322341322,-3362.7367244514226,-1963.5899905296972,-3765.4873003994862,-3261.1390424695064,-976.1835873980768,-1555.3297659662283,-922.9143666233992,-579.5426657303202,-179.1880570768133,233.14935363278727,659.9854132877527,-367.0015509047114,-396.27096464702396,-1434.035978859668,-811.175548330712,-2204.155080929981,-1290.6777179332728,-494.8284400535725,-1046.3767725591092,-644.3229817958731,-400.68691643943674},{246.36608503325584,306.6512430948698,-760.0978406002512,465.7800074784511,760.9086934102233,-546.3055811380217,290.7694629754884,417.17096719703596,553.0758123778282,-513.0549913246477,867.015786011258,626.1754331466758,559.7575119361383,-302.5526002709483,-772.9016274428486,-1496.533693317393,23.14938682149654,-160.55172472868472,-539.6480547265276,-870.8868538313188,-1431.2620012066081,-5919.948435203142,-2115.7360476471595,-21.117794313955187,59.28077035840846,-1012.1681676409406,-1132.7811878558084,-809.0386854367747,-746.4486394578496,1123.9823484136393,-200.1290624207558,1280.4700972102705,724.8879873647029,-882.8391247318846,2717.03732519743,-2614.359456368617,-1690.9772995241888,-1546.2590306885322,-512.3000329697816,668.7839452432889,1435.693739709525,2255.1552159654807,3214.395594701032,-803.6907563407116,-184.45182309833766,417.50416876598376,-778.2734059238728,-546.9094807146968,266.5534996677549,327.3879950831993,-719.6626218851752,-298.91322339079625,1311.9854805051157,2261.179587819281,-276.0987903720305,2182.536859449096,1413.2311348914832,2529.793585017904,472.0586853697656,2850.6883708360156,3142.8887346672827,-835.493951444777,1541.7836225522608,1139.174592914614,-348.911117676318,1210.7585136872644,881.8053934051713,-2102.4342485739753,-796.1724433236585,-1330.4360705341642,-1715.3124209673797,558.7779149293912,3169.0059203881806,711.6053206500845,716.1685328771126,483.52437067668507,-2282.9733102603705,-1186.3779246961549,-611.658686349858,407.1606765800084,-1132.4105939896745,-2252.0203794605504,-714.769935774321,-2370.7765061052737,-5119.975846884576,-1899.434148917532,-668.8423878083711,-201.01378904167174,1754.0249442806053,799.5074300910519,-1631.4971304852138,-1131.6373788138665,1821.5638271916857,2912.711468383038,983.1445541232856,-818.9734207293488,-27.23340524479794,-1925.0749238998976,-50.92007662739947,1967.3690057310464,2226.393231312296,-351.4264215208383,-1618.9066157470659,-1724.7723212520314,387.5395534805663,1451.002908995034,783.0656520491042,-1650.384886623653,-1663.0645094786635,-33.61629792473245,66.10767416361585,864.9388110039895,-1828.0628464486012,1009.0205424349825,-1120.6943057011474,-2669.6333332788836,-1387.9638242776052,-681.8041636183352,-907.9900371736243,-1727.2532201833064,-4154.305696656019,-2930.2752951458897,1698.9186841427495,146.44732066879956,-2377.3580439698985,-2309.264518617056,-627.1844268229414,-2175.869754352706,-1411.5970408689122,1512.7415938799807,-118.85776530388453,-687.3984089411284,-2820.989961910929,-127.0189423202579,80.37688501696869,889.6625314272366,-2472.4272606726495,195.77945735666708,-848.8306873798023,-4219.244548751079,-1842.302938943156,-2408.3299099607925,-667.472154797961,-895.823438528976,167.64641947248452,1578.2270221943465,-52.636854342477925,-1437.227189657209,-2588.229530940911,745.6071221755922,-769.7949034312086,-692.6726484644382,-3709.788230874551,-2659.546848378447,-2845.8619702795368,-440.7258485077821,-633.3848921411709,391.01482480049384,1496.1759393398343,2007.5341013150644,1208.2024720667814,2207.49915867737,975.8557667373625,505.20195661853273,640.4558443639572,-916.6117334705549,1120.8154644553476,-871.099591991601,-978.3293187667512},{-178.08914611537264,620.3348048531336,141.11361764525017,545.1281153676134,741.1741767128606,804.1530574308723,-3157.8513503527142,-4326.009376671033,-2628.6299255797244,-1234.7610724660829,-478.10707100828546,-153.04571510825951,-328.5010900933727,109.92192062582089,-81.53897623009323,292.2027887048313,1106.006526756924,-3021.2198901754805,-1579.9399527212959,-785.621797310445,-1085.8103049252436,-360.77314439014805,-1485.03946538554,-1585.4904335284336,-2774.7856836307365,-125.141626280352,-602.1778950656203,17.19099507751157,-1735.892409026942,-3192.5657018413103,34.08869989586979,531.0667171744175,672.9335871241329,2757.415189475548,-461.6182487047801,1067.1633763732552,994.3484819957229,191.50228553585683,-873.5120461298354,-842.8567191567186,-1864.1065316295833,12.006008671004647,-10.935688187631492,1402.0111063051668,-1461.8012481929388,-375.35492019364904,-296.7154942656273,-1559.521811889219,-828.526436020124,-1655.100044047464,941.6880803652929,-1674.062747541808,34.32397780336812,-478.5971193641772,725.0260462743707,798.3668490042224,1079.1544751998013,2475.7569482635104,-1343.717014851856,-2985.8203360886623,11.3815711592691,808.4261486706838,1854.1745268442767,2680.4797263504097,-1335.900722262549,-1275.8469763357236,-330.65829159023644,1247.0032624250885,5773.793268204731,-3573.044589528694,3182.92882036813,645.9037493360047,38.55576316647454,-707.2932072017369,1748.4507869950667,2415.88221639738,3284.2696321792073,-1019.5341442733296,394.475793701935,-516.1905189938763,-2151.178836732272,-4569.88657149401,-2057.8211359423585,380.086251060321,1177.9417629838306,610.7987005453608,1466.7884194692745,-1493.127124916225,-1797.350036516505,-728.3389377220693,246.87965641876193,212.48646695490206,-1273.573391429374,-2438.922943345403,-2944.6895835441933,117.14655151102211,739.5232173228193,-585.6782151151348,-666.1532806780882,-1009.3501268196617,-3297.8481400674427,105.04921500857564,-3692.654678456764,724.388861193575,-579.3658618535007,-2651.327205925901,-668.2241249211389,1509.2255960171376,1610.5737720367558,4216.278261074874,-3467.8337125821804,790.52353331018,-1032.7155857359478,-652.8159895888163,52.166382253816366,-432.26370853692686,-2149.428145056218,-390.5172766709842,-2725.342588980063,172.4797914374315,770.7488690433447,-17.60168079530382,-1255.824799340617,-2022.5303384271767,-1724.3197977489021,-772.5523698129114,-944.4530914427922,-647.9972263634761,-576.4875202716112,-368.4820515837454,-45.030647295521845,-2699.5921674208935,1947.7872137518586,-628.407020079288,-2139.8625243258616,-1076.120185781626,143.76816787255075,74.75093261988575,-374.8239147733235,3621.834800336144,218.5555635779876,397.3128545928288,459.0869029659165,-175.7675820945712,-1462.304813280248,-3655.8385785651817,992.2122699551537,-1118.879007071171,2644.2626995737314,2225.3891023304964,443.50735843253864,-2351.1632021324167,800.8818784801795,-2788.7506880892283,162.11439407064358,149.2297341736649,9.11193766349582,-1090.4746174822085,-1371.0547173487655,-3862.5537861652015,-3726.5602599236145,-3942.486113583327,-5115.759035301664,-2461.6110766226643,-158.7471121898222,-68.54783424415069,-270.0350104708497,-596.9011578458427,387.6101600984217},{759.1859690708278,-278.1728794483993,-441.95678621977993,-188.3912969973454,919.3042824187262,-184.3948443580472,-1040.0631513308185,-1406.4086255071782,-1037.1667605623154,82.44888600680916,-568.677423486329,259.0702068586904,-658.2565537522668,550.8097107871571,-646.3784075873706,457.7524075444026,-1908.6782763311494,-1589.9328294126876,943.6498504829469,-4010.867629127782,-5512.021598582396,-4312.064150607006,-5820.755195809004,-3572.290606464536,-566.8312090238331,-297.22176176429934,-817.1017338598135,-1254.3006267772098,-3208.4834998113274,-1889.12462686684,1678.7047924973297,-1662.5957097084647,1078.3829300632874,3282.195871654824,836.4340586759143,335.7569316823713,67.48393119683338,-4234.1201084902805,-2391.8200547203824,-160.76847819574084,-2243.010616018957,-1624.87995249134,-1433.1426561561493,-1403.1623576191105,658.0288363841082,3326.1097325121,2014.8202015304385,236.84986530364017,-717.1200642716548,-1332.652081687473,-3696.724915906863,-3994.0879832852174,-70.91909336292451,-4383.163580617164,942.6916936420813,-991.5792904171417,112.57837149608338,2.6422391708929722,-731.7238667509279,-349.86601084356556,-3251.6426473304005,-343.1641260102865,68.56888881692086,1093.5748800425476,-2238.0849384714693,-1391.1579040919887,40.149480287671906,3005.825957548617,2288.6010080919614,1418.3015362340307,1650.7005582044567,-2390.4069920087163,-644.9916510082068,612.0503721410092,2674.1792540591405,709.7787001611769,341.847825024601,141.85645131416254,921.830736405611,52.73703517333652,1195.4306811077463,410.87625476770586,-184.10537249418425,1366.847632526371,2008.3466945397447,2149.6994060825637,-1111.1840180954925,-998.2723919781193,290.241924199538,-2186.318494071098,-1512.711768427388,-213.83890663323123,1409.956151374588,-1303.705447167383,-851.5420635555598,1498.1893054753302,565.5317184821738,-4385.310740858516,-957.3171635720562,864.9844134963848,-1425.317376551192,476.3815569582154,-2445.913764222115,-1939.9408124315166,-254.93714982124317,-1466.197009492118,-247.45941646528343,69.67737223921623,916.4921862092965,-164.21035994786425,1773.3628435263076,-2131.3724911797453,1766.5236510613786,-1178.4891272233776,-573.9476017798277,-2520.152612055466,-1863.9117434421878,116.70857793086206,-614.5939713023243,-720.9119669927295,-1365.3152790131885,-2097.6626842998185,-826.6588238392949,-1899.9686762755243,-1887.089110634138,-1785.6166259455667,-425.8176083053186,-989.833343010999,64.33991050453953,1202.2738809117486,-83.42350988089632,-495.4501307924721,-1250.2006212021683,966.9270307340856,-369.5407580319568,1058.5724236857432,128.43751308381596,-242.40363353465804,-1693.3926374979083,11.235425557537233,-807.7367640310125,-630.0597215203126,177.87951430397956,-994.6036366517611,-452.7232520407279,-1150.2602862731044,-20.667450513120624,-1017.9300111135452,-1808.5117512396096,-3595.153948004449,-167.29948063516997,-2722.2348601120284,-758.5689868974157,1886.798241752372,2051.314478192683,-130.9625732593833,714.1698928827154,1107.5945671919658,-1020.6328128814299,1053.5408356361781,-688.684052556394,854.0731062426393,1473.6964317763716,-122.76151324067838,2236.6147478269386,2730.844196365826,-45.71810605352407,670.996928504671,967.50502435832}};
float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void setup() {
  Serial.begin(9600);
  pinMode(DATA_LED,OUTPUT);
  pinMode(SUCCESS_LED,OUTPUT);
  Serial.println("connected successfully");
  lcd.begin(16,2);

  while (!Serial.available()) {
    lcd.setCursor (0,0);
    lcd.print("   Waiting..   ");
    lcd.setCursor (0,1);
    lcd.print("       :)     ");
  
    digitalWrite(DATA_LED,HIGH);
    delay(100);
    digitalWrite(DATA_LED,LOW);
    delay(100);
    }
  
  lcd.clear(); 
  
}
bool is_image_sent = false;
int row = 0;
int column = 0;
void loop() {
    
     while (Serial.available() > 0) {
      lcd.setCursor(0,0);
      lcd.print("Receiving Pixels..");
      digitalWrite(DATA_LED,HIGH);
       int pixel = Serial.parseInt();
       if (Serial.read() =='\n') {
        count++;
        input_image_square[row][column] = pixel;
        update_row_column();
        lcd.setCursor(0,1);
        lcd.print("       ");
        lcd.print(pixel);
        lcd.print("       ");
        Serial.println(pixel);
        }
        if (count==784) {
          is_image_sent = true;
          count = 0;
          digitalWrite(DATA_LED,LOW);
          digitalWrite(SUCCESS_LED,HIGH);
          }
  }
  if (is_image_sent) {

    classify();
    is_image_sent = false;
    digitalWrite(SUCCESS_LED,LOW);
  }
}

void update_row_column() {
  if (column == 27) {
    row++;
    column = 0;
  } else {
    column++;  
  }

}


void classify() {
  lcd.clear();
  lcd.print("Classifiying..");

  convolve_image();
  max_pooling();
  forward_propagation();
  print_output_vector();
  
  int winning_class = find_winning_class();
  
  Serial.print("winning class:");
  Serial.println(winning_class);
    

    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Winning Class Is: ");
    lcd.setCursor(0,1);
    lcd.print(winning_class);
    lcd.print("    ");
    lcd.print(output[winning_class]*100);
    lcd.print("%");

    delay(2000);
}

void convolve_image() {
    
  int compressed = 0;
  int nonCompressed = 0;

 
      
  for (int i=0;i<CONVOLVED_IMAGE_SIZE;i++) {
    
    for (int j=0;j<CONVOLVED_IMAGE_SIZE;j++) {
      float sum = 0;

      for (int k=0;k<KERNEL_SIZE;k++) {
        for (int l=0;l<KERNEL_SIZE;l++) {
          if (kernel[k][l]>=0.4 || input_image_square[i+k][j+l]/255.0 > 0.0) {
              sum += input_image_square[i+k][j+l]/255.0 * kernel[k][l]  ;
              nonCompressed++;
            } else {
              compressed++;
              }
        }
      }
      convolved_image[i][j] = sum * PIXEL_PRECISION;

      
      
      }

  }

}

void max_pooling() {
  
      for (int i = 0; i < CONVOLVED_IMAGE_SIZE; i += STRIDE) {
        for (int j = 0; j < CONVOLVED_IMAGE_SIZE; j += STRIDE) {
            int max_val = convolved_image[i][j];
            for (int m = i; m < i + STRIDE; m++) {
                for (int n = j; n < j + STRIDE; n++) {
                    if (convolved_image[m][n] > max_val) {
                        max_val = convolved_image[m][n];
                    }
                }
            }
            pool_output[i/STRIDE][j/STRIDE] = max_val;
        }
    }

  }

void forward_propagation() {
  // Reshape the 2D array to a 1D array for input to the neural network
  int flattened_input[POOL_SIZE* POOL_SIZE];
  int k = 0;
  for (int i = 0; i < POOL_SIZE; i++) {
    for (int j = 0; j < POOL_SIZE; j++) {
      flattened_input[k] = pool_output[i][j];
      k++;
    }
  }


  // Feed the flattened input to the fully connected neural network
  for (int i = 0; i < N_OUTPUT; i++) {
    float sum = 0;
    for (int j = 0; j < N_INPUT; j++) {
      sum += (flattened_input[j]/PIXEL_PRECISION) * (output_weights[i][j]/WEIGHT_PRECISION) ;
    }
    //  lcd.clear();

      //    lcd.print(sum);
      //delay(1000);
    output[i] = sigmoid(sum);
  }    
    
}

void print_output_vector() {

  Serial.print("output vector: ");
  for (int i=0;i<N_OUTPUT;i++) {
    Serial.print(output[i]);
    Serial.print(" ");
  }  
  Serial.println();
}

int find_winning_class() {
  
  float maximumOutput = output[0];
  int winningClass = 0;
  
  for (int i = 1; i< N_OUTPUT;i++) {
    if (output[i] > maximumOutput) {
        maximumOutput= output[i];
        winningClass = i;
      }
  }
  return winningClass;  
}
