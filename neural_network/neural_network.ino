#include <math.h>
#include <stdlib.h>


const int IMAGE_SIZE = 28; 
const int N_OUTPUT = 10;      // 10 classes (digits 0-9)
const int KERNEL_SIZE = 3;
const int STRIDE = 4;
const int CONVOLVED_IMAGE_SIZE = IMAGE_SIZE-KERNEL_SIZE+1;
const int POOL_SIZE = (CONVOLVED_IMAGE_SIZE-1)/STRIDE + 1; 
const int N_INPUT = POOL_SIZE * POOL_SIZE;  // input neurouns


float kernel[KERNEL_SIZE][KERNEL_SIZE]= {{0.05,0.3,0.001},{0.024,0.104,0.53}, {0.21,0.59,0.98}}; //random kernel

int input_image_square[IMAGE_SIZE][IMAGE_SIZE] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.0, 192.0, 134.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 77.0, 5.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 235.0, 250.0, 169.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 220.0, 241.0, 37.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 189.0, 253.0, 147.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 139.0, 253.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0, 253.0, 253.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 43.0, 254.0, 173.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.0, 153.0, 253.0, 96.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 43.0, 231.0, 254.0, 92.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 163.0, 255.0, 204.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 104.0, 254.0, 158.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 162.0, 253.0, 178.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 131.0, 237.0, 253.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 162.0, 253.0, 253.0, 191.0, 175.0, 70.0, 70.0, 70.0, 70.0, 133.0, 197.0, 253.0, 253.0, 169.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 51.0, 228.0, 253.0, 253.0, 254.0, 253.0, 253.0, 253.0, 253.0, 254.0, 253.0, 253.0, 219.0, 35.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 65.0, 137.0, 254.0, 232.0, 137.0, 137.0, 137.0, 44.0, 253.0, 253.0, 161.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.0, 254.0, 206.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 160.0, 253.0, 69.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 85.0, 254.0, 241.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 158.0, 254.0, 165.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 231.0, 244.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 104.0, 254.0, 232.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 208.0, 253.0, 157.0, 0.0, 13.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 208.0, 253.0, 154.0, 91.0, 204.0, 161.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 208.0, 253.0, 254.0, 253.0, 154.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.0, 190.0, 128.0, 23.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}; // pixels of class 4

float pool_output[POOL_SIZE][POOL_SIZE];

// trained output_weights
float output_weights[POOL_SIZE*POOL_SIZE][N_OUTPUT] = {{0.24171661593276342,-0.16555587737336444,-1.1589937222897329,-1.1641490323446861,0.32514443493910683,-0.1846419569977924,0.9129483660902981,-1.0096155396502144,-1.2691797373074114,0.6231045400324393},{-2.4473737001861,1.7025422185201284,-1.2986142737470752,-3.1491845021565665,-0.6640824521044048,-1.5999530368597936,0.6284689527431975,-1.6386826349403116,-5.023633143251388,-1.7221247511320645},{-2.0272986334691994,-0.04784430812990966,-1.1798001525340884,-0.8031576695235733,-2.132027158634921,-0.3120693143942851,1.6478160516111784,-2.2096640732065262,-3.9961474377395394,-2.652604360921749},{-0.20795181330447585,-1.4551081503430559,1.3946338891887422,0.31365342081744885,-1.051055636820847,-2.757256292405763,1.677383432517161,-2.0870057731592064,4.210597659064863,-5.759972218023591},{-1.3963188778830575,0.3439327309937357,-2.2923170688844507,-2.9455733950060883,-0.7465099945960884,-1.7625744080044974,8.833672698535425,-6.479244884985322,-0.5377073142971396,-7.740353570927825},{-2.2807423205956128,1.3498634131814813,2.2843332076910188,-6.509602651143067,0.7526796939763732,-2.8998329831805023,4.293192812537272,-2.10323990144684,1.890935530776169,-5.523388174851258},{-1.3833111924307162,-0.14384674635545452,0.6933876186490141,-1.4279018851177159,0.6792706093997625,-0.8680699081282712,-0.19170330215917092,-0.635120858068717,-0.3304212452561762,-1.4542592054852599},{-1.7420344460534738,-1.0787475717175465,2.348787367371853,3.7737862488887615,1.2362766943225068,-3.360712119997161,-0.615968781003401,-0.08881521069312222,-2.668665595542086,-5.8287250376517665},{-1.7677186789256734,-3.10193645873929,2.216958679766339,3.3575210109505287,1.0346552122618828,-0.5339592559319891,-1.6752988194585028,2.8757902237431314,1.585305464130452,-4.26462585945812},{-0.4175058748177282,-5.384557402708543,3.681005224772411,-0.0382780437001564,-0.9654094641190082,-1.653809616978851,-3.8553750835577696,-0.31024856846401794,2.87379892581798,0.6186392352233778},{2.2068196005739833,-1.2996710577506743,2.155145017239776,-4.104866917626584,-4.442316724605384,-0.7282338968719896,-2.6556249038215665,-0.7140899342952794,-2.2352336824876398,5.4506733134498795},{0.8040324182563809,-4.316750888671273,0.4401531371612339,0.10357125519302744,0.7260453395276374,0.34993325435556555,-7.463599215365111,-0.13861317465185133,1.128284637528753,-0.7772728944834528},{0.8415341357778227,-2.6545349319231932,-1.5992019165192406,-1.0269338613875576,5.024025760180423,5.574254408535138,-4.582481123102907,0.28646127778139924,-1.1648601752613608,-5.882998840828167},{-5.605691499017407,-2.8041490342778204,-0.9221520548160969,-6.064332665670877,-0.6559259750509151,8.041396951111322,0.5184950349451719,-1.7974313338352914,-1.3557320337724603,-6.459939011059785},{-0.9216273115321035,-0.7324096271697771,0.38443550173762664,-1.0225996705735116,-2.951680092123558,-1.3841898923740612,-1.0398949796555537,2.7420602775628033,1.0887183488074477,0.03886163798842447},{-1.1378105662523907,-0.271919184453661,-5.553028808322128,-4.455247350020891,-0.9849718485361693,1.072397750006471,0.02182632290512445,0.8695240617123566,-2.0340159941828553,2.030186801497386},{0.30363713264529246,-2.895485837262925,-5.386605341603265,-1.3015768957244274,2.1004357262720603,8.380549859912877,4.466404150878342,-0.2775322730949403,2.278159774083092,-1.4979443673648132},{-1.0242118853797333,7.990797741446091,-6.477322720525407,4.334231440255412,-2.1188009650119506,-2.937222005126035,-2.092642745584012,4.121781981926204,-2.944783119069938,-3.9785467762468523},{0.5269344062087014,0.09754035151195169,1.0037831236850823,0.9796318607750955,1.0169725315723417,-2.7481889001323787,-3.4098906963022677,2.570611795001508,0.6959555690914278,-0.7723412144956396},{2.6076667649922434,-5.055206927878398,0.0718537687073395,-2.7314729631053525,-4.82733129652197,-7.527877353932642,2.3216107082861153,-2.165525146829798,7.070202652840369,3.748418490100587},{-1.588930303374054,-1.9684055612651032,-1.8377910723489665,-1.7257646477466553,-2.808692494229375,6.402724951129758,0.25292793568053656,-3.3148508835173462,-1.7573916870677129,-3.1891373868603297},{1.449274639914052,-0.09615849941074742,-3.3977231716218035,-3.3522450827140586,0.962739558579963,-2.561744633938309,-2.4784669157238923,-0.35740863759627844,-3.3259881433970704,0.3461823744664098},{3.4493345094687196,-4.989223215390841,0.8413343612533579,-0.6381862294412722,3.745581607625568,-1.8974293106133682,1.0215395779430063,-2.414378906166608,-2.1807984571519388,1.171411771562242},{0.785959540872272,-0.43615165194549116,-0.934342885145614,0.4519146822818175,2.7558868314568583,-0.9333574027509189,-1.1272235943140438,-5.372476376572905,-0.17635110391925873,2.8789360371303614},{-12.840384794388687,2.463603761703691,-0.05332692859733444,0.5093093063586008,1.359335296776889,-0.9523609703857137,0.5017290373968438,-4.913447541183173,5.3282175382020895,-0.1665495906778891},{1.4098536188364135,-4.7454909247098644,-1.9962882447995547,-1.1884542089612675,3.2151953095383705,-1.9753913305984683,-0.6021739804049487,1.8011364807349723,-2.962612578299687,3.573698204391277},{1.2530437258344316,-2.1531063844019,-5.044827461697286,-1.014528972673658,3.899572804265094,-2.8253896752414005,4.779243663750538,3.651661000217874,-1.8007009166343564,-3.3594001995880047},{1.9491765706610606,0.5623729841981575,1.3650150107141288,-0.7498992355819596,-0.5350604224035197,-3.5962381853478087,-2.220847101465668,-2.918374988689383,0.40735629059489803,-2.5197131725140673},{-2.84721821882519,-1.5570876478424587,1.8949120636165702,5.487599351426412,-4.018014865537869,0.7845795049185504,-6.557077960386414,-0.19583300072814452,-3.3363213015764956,-2.272404295901387},{0.3967121499134796,-1.977808160079287,0.3658382927783921,-0.03166215135984658,-1.24608680132063,0.9271605947893921,0.2770458561837595,-0.6694696563838872,-0.1370082792288904,-0.5680501030308642},{2.668744106588891,-0.8571648522816435,3.2800471478330704,-6.633389560640291,-1.5195454926256575,-3.367811019767883,5.0809248763612445,3.170074533428139,2.555493913598387,-0.956793892629639},{-6.216167182800216,3.288356349705257,1.331841492564457,-6.018530890652108,0.8912994597367525,-2.356620549833608,-2.7966072758849223,-0.8570441893739555,-9.608319926930372,-3.279315834596077},{1.9337554139129132,-5.4273707633837995,1.8342798582610318,4.180259684730035,-3.7735073593057105,1.7510810645689778,3.406926680500551,0.3403635577534095,-1.9439591454813339,-1.1723050007696145},{-3.3076748600670753,-0.6796535622298336,6.116472671874599,-1.4350578208693083,-5.438112477162458,2.244198152057902,-2.600053497137986,-5.257656392943084,-4.111303427040053,-2.677495617371294},{0.3861530671612422,1.1283415974678763,4.222680567550878,-4.0082642880012695,-0.5272887558978072,-2.745444274491184,-2.5500194810121197,-1.6321294792031271,0.45835740452869994,-1.988152403943557},{0.9392906207421771,-0.5898268571802617,-0.6049409584212916,2.551167011078543,-2.54125967481243,-2.344081766003939,-1.1309298527387688,0.3855681722772418,-1.032986889349383,-0.46959126085736635},{0.5077582176332609,2.40273426619087,1.291555698705079,3.253547272605386,-5.852012035963773,1.3334402809929464,-4.601065773529552,-3.154620860863054,0.7971658895416047,-0.4471468762701546},{4.442653899328102,-0.46194656483424695,-1.3024595018612115,3.15479347696186,-4.087271206026415,3.6021156822013194,0.9553950160630811,-2.0313857359438874,-1.6067500555241476,0.6894206340022334},{0.7205547602027389,-0.9283718587144573,-3.8020220280460326,3.6658557554856888,-0.22136060769530205,-1.6204893977769523,-1.729338850942878,-0.018050538716623453,2.0978569426731917,-1.687256228575386},{-2.4027910147854197,2.599003314789219,-1.3662768770131115,-1.5709311628207916,0.7823426249224721,1.7576976031632516,-0.2544101868085784,-4.2650311172968305,1.5255574342077076,-1.217442358165317},{0.029720047180196992,-1.219285856826548,-1.1174036600283113,0.5502079362781416,1.8836385486299425,-0.5003246434077687,-3.1897776654249568,-3.068390203947226,1.5059556043033435,2.6002415659572145},{-0.7563284926368102,0.8503264948158001,0.9603240068531946,-2.0989973873727035,-0.8590025441038597,-1.6779171571348057,-1.2129726744434473,0.882232632056585,0.06829754669306466,1.374935135401169},{0.026675274178864096,-0.4083005267919326,-0.09509494179088432,3.0627541285676725,0.07921645572908385,-1.4405715456516264,-0.18999925053179376,1.546737462976388,-2.1397645987665395,-0.36086079024274453},{-3.772742782376543,-0.7912947012006694,-4.791492147084693,3.1797585509806856,0.29951398285376074,-3.0109821494654496,-1.1802388109597204,5.907317898300808,-5.416850325464231,0.6289893915071876},{-3.4645306239370135,-3.254168129524504,-3.8746957194685288,-1.4949875943956408,-2.363044443222875,-1.6693327428596312,-3.783694920593727,2.28774265364338,-0.49121832940488164,-0.583130421068728},{-3.667706328076925,-2.840656224842326,-0.6341995040610209,-4.535485164196693,-5.938800307165998,-0.4447373958130078,-4.858285043502176,2.4811821603489776,-0.20794103301877753,0.5283485673545293},{-3.8028137557232644,-0.5227387985203665,1.490224258538022,-2.0681919861128715,-6.012935123604341,-2.7205167433687953,-4.195529287818847,1.3193800349131013,-0.47922956616466944,2.70795745641628},{-1.3499960969811395,-0.0331362568421897,-1.5754200069897053,0.8236781970427121,-1.3499280801340612,-1.9628027174815912,-1.4385624076130938,0.5465256507050112,-0.8460535764151522,1.819955721823947},{-0.9138486498231323,-0.2831541183872443,0.6739789338117687,0.6738817339655072,-0.08052129241127659,-0.8852240473213524,0.8207060153351489,-0.866579103438774,0.5049496053941936,0.4526719569105391}};

float convolved_image[CONVOLVED_IMAGE_SIZE][CONVOLVED_IMAGE_SIZE];

float output[N_OUTPUT] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

  
// sigmoid activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}


void setup() {
  Serial.begin(9600);
}

void loop() {

  // convolve the input image
    Serial.println("convolution!!!");
  for (int i=0;i<CONVOLVED_IMAGE_SIZE;i++) {
    
    //Serial.println("hello there, I'm convolving and there's no problem");
        
    for (int j=0;j<CONVOLVED_IMAGE_SIZE;j++) {
      float sum = 0;
      for (int k=0;k<KERNEL_SIZE;k++) {
        for (int l=0;l<KERNEL_SIZE;l++) {
          sum += input_image_square[i+k][j+l]/255.0 * kernel[k][l];
        }
      }
      convolved_image[i][j] = sum;
      }
  }
  Serial.println(F("max pooling!"));

  // implement max pooling
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

      free(convolved_image);

  // Reshape the 2D array to a 1D array for input to the neural network
  float flattened_input[POOL_SIZE* POOL_SIZE];
  int k = 0;
  for (int i = 0; i < POOL_SIZE; i++) {
    for (int j = 0; j < POOL_SIZE; j++) {
      flattened_input[k] = pool_output[i][j]/255.0;
      k++;
    }
  }

  Serial.print("input image to neural network with size of:");
  Serial.println(POOL_SIZE*POOL_SIZE);
  
  for (int i =0;i<POOL_SIZE*POOL_SIZE;i++) {
      Serial.print(flattened_input[i]);
      Serial.print(" ");
    }

    Serial.println("\n\n\n\n");
  // Feed the flattened input to the fully connected neural network
  for (int i = 0; i < N_OUTPUT; i++) {
    float sum = 0;
    for (int j = 0; j < N_INPUT; j++) {
      sum += flattened_input[j] * output_weights[j][i];
    }
    output[i] = sigmoid(sum);
  }
  

  float maximumOutput = output[0];
  int winningClass = 0;

  Serial.print("output neurons:");
  for (int i=0;i<N_OUTPUT;i++) {
    Serial.print(output[i]);
    Serial.print(" ");
  }
  
  Serial.print("\nand we have a winner!!:");
  
  for (int i = 1; i< N_OUTPUT;i++) {
    if (output[i] > maximumOutput) {
        maximumOutput= output[i];
        winningClass = i;
      }
  }
      Serial.println(winningClass);
        free(pool_output);
        delay(2000);
}
