### v1.0.4.rc1 main updates

1. Fix for a deprecated PolynomialFeatures method for newer sklearn versions
2. Linear SEM now supports a list of noise types
3. Support for networkx versions 2.8+
4. Add GES to inline algorithms

### v1.0.3 main updates

1. Add new algorithms: DAG-GNN, GES and PNL.
2. Pytorch implementation for GAE algorithm.
3. PC algorithm supporting prior knowledge injection.
4. Test code for all implemented algorithms.

### v1.0.3rc3 main updates

1. Fix a GPU bug in the RL method.
2. Removed the 'config' input parameter and replaced it with the actual variable names.
2. Using the 'workflow_dispatch' trigger for CIT. 

### v1.0.3rc2 main updates

1. Removed the 3rd party dependencies: loguru, python-igraph
2. Updated the CI test methods, including：
CITest.gauss
CITest.g2_test
CITest.chi2_test
CITest.freeman_tukey
CITest.modify_log_likelihood
CITest.neyman
CITest.cressie_read
3. Added the algorithm: ANM
4. Rename 'GraN_DAG' to GraNDAG
