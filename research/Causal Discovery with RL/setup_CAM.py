import rpy2.robjects.packages as rpackages

utils = rpackages.importr('utils')

pck_names = ['Matrix', 'mgcv', 'codetools', 'rpart', 'glmnet', 'mboost', 'CAM']

for pck_name in pck_names:
    if rpackages.isinstalled(pck_name):
        continue
    if pck_name != 'CAM':
        utils.install_packages(pck_name)
    else:
        #install CAM locally
        utils.install_packages('CAM_1.0.tar.gz', type='source')
        print('R packages CAM has been successfully installed.')

# check if CAM and mboost have been installed
if rpackages.isinstalled('CAM') and rpackages.isinstalled('mboost'):
    print('R packages CAM and mboost have been installed')
else:
    print('need to install CAM and mboost')

