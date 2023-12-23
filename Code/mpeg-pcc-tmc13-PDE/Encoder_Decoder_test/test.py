import os
import shutil
#需要修改的参数

current_fold=os.getcwd()+"\\"
tmc3 = current_fold+'tmc3.exe'
pce = current_fold+'pc_error.exe'
#os.mkdir('log')
		

def getResolution(cfgpath):
	pcerrorcfg = (cfgpath + 'pcerror.cfg')
	reader = open( pcerrorcfg, 'r')
	Resolution = 0 
	for line in reader:
		words = line.split()
		if ('resolution:' == words[0]):
			Resolution = int(words[1])
			break
	return str(Resolution)

def getPointCount(dec,declog,enclog):
	decfile = open(dec,encoding="gbk",errors='ignore')
	voxelnum = 0
	for line in decfile:
		words = line.split()
		if words[0]=='element' and words[1]=='vertex':
			voxelnum = words[2]
			break
	changedeclog  = open(enclog,'a')
	changedeclog.write('\nTotal point count: ' + voxelnum)
	changedeclog  = open(declog,'a')
	changedeclog.write('\nTotal point count: ' + voxelnum)

data_list = list([
	'basketball_player_vox11_00000200',
	'dancer_vox11_00000001',
	'facade_00064_vox11',
	'longdress_vox10_1300',
	'loot_vox10_1200',
	'queen_0200',
	'redandblack_vox10_1550',
	'soldier_vox10_0690',
	'facade_00009_vox12',
	'shiva_00035_vox20',
	'staue_klimt_vox12',
	'egyptian_mask_vox12'
])
def runEED(data_list):
	# EEDconfig = ' --EEDsigma=' + EEDsigma + ' --EEDdistance=' + EEDdistance + \
    #     ' --EEDlambda=' + EEDlambda + ' --EEDseedweight=' + EEDseedweight+' --EEDSeedQpOffset='+'6'+' --EEDWeightOffset='+'4'
	for pointCLoudname in data_list:
		pointCLoudname='\\'+pointCLoudname
		encconfig =(current_fold+ 'encoder.cfg')
		decconfig =(current_fold + 'decoder.cfg')
		seq = (current_fold+"pointclouds\\" +pointCLoudname +'.ply')
		bin = (current_fold+'midfiles' + pointCLoudname + '.bin')
		# pointCLoudname=pointCLoudname+'_'+str(EEDsigma)+'_'+str(EEDdistance)+'_'+str(EEDlambda)+'_'+str(EEDseedweight)
		# enc = (current_fold+'midfiles' + pointCLoudname + '_enc.ply')
		res = (current_fold+'midfiles' + pointCLoudname + '_res.ply')
		dec = (current_fold+'midfiles' + pointCLoudname + '_dec.ply')
		enclog = (current_fold+'log' + pointCLoudname+'_enc.log')
		declog = (current_fold+'log' + pointCLoudname+'_dec.log')
		pcelog = (current_fold+'log' + pointCLoudname+'_pce.log')
		os.system(tmc3 + ' --config=' + encconfig + ' --uncompressedDataPath=' + seq + ' --reconstructedDataPath='  + res + ' --compressedStreamPath='  + bin + ' >' + enclog) 
		os.system(tmc3 + ' --config=' + decconfig +' --uncompressedDataPath=' + seq + ' --reconstructedDataPath='  + dec + ' --compressedStreamPath=' + bin + ' >' + declog) 
		Resolution = getResolution(current_fold)
		os.system(pce + ' -a ' + seq + ' -b ' + dec + ' -c  -r '+ Resolution + ' >' + pcelog)
		getPointCount(dec,declog,enclog)
		# print(' --config=' + encconfig+EEDconfig + ' --uncompressedDataPath=' + seq + ' --reconstructedDataPath='  + res + ' --compressedStreamPath='  + bin)
		# print(' --config=' + decconfig +' --uncompressedDataPath=' + seq + ' --reconstructedDataPath='  + dec + ' --compressedStreamPath=' + bin)
		# print(' -a ' + seq + ' -b ' + dec + ' -c  -r '+ Resolution + ' >' + pcelog)
		#运行end


# EEDsigma='300'
# EEDdistance='50'
# EEDlambda='80'
# EEDseedweight='6'
runEED(['longdress_vox10_1300'])


		
