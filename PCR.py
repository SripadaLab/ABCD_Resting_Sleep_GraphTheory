import sklearn
from sklearn import decomposition
from scipy.linalg import lstsq, pinv


def OLS(X, y):
    """
    Performs Ordinary Least Squares (OLS) regression to compute the regression coefficients.

    Parameters:
    -----------
    X : numpy.ndarray
        The design matrix or independent variables (features).
    
    y : numpy.ndarray
        The dependent variable or response vector.
    
    Returns:
    --------
    numpy.ndarray
        The computed OLS regression coefficients.
    """

    return lstsq(X, y, lapack_driver='gelsy', check_finite=False)[0]


def covariate_residualization(X, covars, train_idxs):
    """
    Removes the effect of covariates from the data by calculating the residuals after regressing out covariates.
    The covariates are removed independently for each column in the data matrix X.

    Parameters:
    -----------
    X : numpy.ndarray
        The data matrix where the effect of covariates will be removed. Each column in X is treated independently.
    
    covars : numpy.ndarray
        The covariate matrix used for residualization.
    
    train_idxs : array-like
        Indices corresponding to the training data used for fitting the model.
    
    Returns:
    --------
    numpy.ndarray
        The residualized data matrix, with covariate effects removed independently for each column in X.
    """

    betas = pinv(covars[train_idxs, :]) @ X[train_idxs, :]
    return X - (covars @ betas)


def PCR(X, y, covariates, fold_structure, n_nested_cv=5, n_pcs_consider=None):
    """
    The Brain Basis Set (BBS) predictive model was first discussed in:
    Sripada, C. et al. Basic Units of Inter-Individual Variation in Resting State Connectomes; Sci Rep 9, 1900 (2019); https://doi.org/10.1038/s41598-018-38406-5
    
    This approach is similar to principal component regression with an added predictive modeling element. In a training partition, we calculate the expression scores for each of k components for each subject by projecting each subject’s connectivity matrix onto each component. We then fit a linear regression model with these expression scores as predictors and the phenotype of interest as the outcome, saving B, the k × 1 vector of fitted coefficients, for later use. In a test partition, we again calculate the expression scores for each of the k components for each subject. Our predicted phenotype for each test subject is the dot product of B learned from the training partition with the vector of component expression scores for that subject.
    
    Cross-validation is used to assess model generalizability, and nested k-fold cross-validation is used for hyperparameter tuning. There is a single hyperparameter in this model, the number of PCA components used to fit the linear regression model. Covariates are accounted for using the partial regression methodology - ie: compute the residuals of regressing the covariates against the predictors (expression scores) and response (phenotype) and fit the linear regression model using these residuals.
        
    
    Parameters
    ----------
    X : ndarray
        2-dimensional numpy array containing connectivity data for each subject. Shape must be (number_of_subjects, number_of_edges)
        Each row is the flattened upper-triangular matrix of pearson connectomes per subject
        
    y : ndarray
        2-dimensional numpy array containing the phenotypic data for each subject. Shape must be (number_of_subjects, 1)

    covariates : ndarray
        2-dimensional numpy array containing the covariate/nuisance data for each subject. Shape must be (number_of_subjects, number_of_covariates)
        
    fold_structure: list
        List of 2-tuples, where each tuple t_i is (training indices, held-out test indices) for cross-validation fold i. Length must be (number_of_cross_validation_folds)
        
    n_nested_cv: int
        Number of nested cross-validation folds used for hyperparameter tuning

    n_pcs_consider: list
        List of hyperparameter values (number of components) evaluated in model tuning
         
       
    Returns
    -------
    list:
         List of length (number_of_cross_validation_folds) that contains the predictive performance for each held-out test cross validation fold  
         
    list:
         List of length (number_of_cross_validation_folds) that contains the number of PCs selected by nested cross-validation for each cross validation fold  
    """
    
    
    np.random.seed(42)                  # set numpy RandomState for reproducibility
    test_rs = []                        # list of length (number_of_cross_validation_folds) that will be populated with the predictive performance for each CV fold
    n_pcs_selected = []                 # list of length (number_of_cross_validation_folds) that will be populated with the number of PCs selected by nested cross-validation for each CV fold
    if n_pcs_consider is None:          # list of number of principal components evaluated for predictive modeling (higher means longer runtime)
        n_pcs_consider = np.arange(min(X.shape))
    
    for train_idxs, test_idxs in fold_structure:
        # shuffle train_idxs and chunk into held-oout fold structure for nested cross validation
        valid_folds = np.array_split(np.random.choice(train_idxs, size=len(train_idxs), replace=False), n_nested_cv)
        # init matrix used to store nested cross validation results
        valid_perf = np.zeros((n_nested_cv, len(n_pcs_consider)))
        # start of nested CV
        for valid_i, valid_idxs in enumerate(valid_folds):
            train_idxs_subset = np.array([x for x in train_idxs if x not in valid_idxs])
            # fit PCA on training data, and evaluate number of PCs to use for predictive modeling using nested-CV
            pca_model_edges = decomposition.PCA(n_components=n_pcs_consider[-1], random_state=42).fit(X[train_idxs_subset, :])
            X_transform = pca_model_edges.transform(X)
            # residualize predictors of covariates
            X_transform = covariate_residualization(X_transform, covariates, train_idxs_subset)
            # evaluate held-out performance in nested CV folds for increasing number of PCs
            for pc_i, n_pcs in enumerate(tqdm(n_pcs_consider, ncols=1000, leave=False)):
                nested_X_transform = X_transform[:, :n_pcs]
                y_cov_controlled = covariate_residualization(y, covariates, train_idxs_subset).flatten()
                betas = OLS(nested_X_transform[train_idxs_subset, :], y_cov_controlled[train_idxs_subset])
                y_preds = (nested_X_transform @ betas.reshape(-1, 1)).flatten()
                valid_perf[valid_i, pc_i] = stats.pearsonr(y_cov_controlled[valid_idxs], y_preds[valid_idxs])[0]
                
        # determine number of PCs that yielded the highest nested-CV predicitivity, averaging across nested-CV folds  
        mu_valid_perf = valid_perf.mean(0)
        opt_valid_perf_idx = np.argwhere(mu_valid_perf == np.max(mu_valid_perf)).flatten()
        opt_edges_pcs = n_pcs_consider[opt_valid_perf_idx[0]]
        
        pca_model_edges = decomposition.PCA(n_components=int(opt_edges_pcs), random_state=42).fit(X[train_idxs, :])
        X_transform = pca_model_edges.transform(X)
        X_transform = covariate_residualization(X_transform, covariates, train_idxs)
        y_cov_controlled = covariate_residualization(y, covariates, train_idxs).flatten()
        betas = OLS(X_transform[train_idxs, :], y_cov_controlled[train_idxs])
        y_preds = (X_transform @ betas.reshape(-1, 1)).flatten()

        test_rs.append(stats.pearsonr(y_cov_controlled[test_idxs], y_preds[test_idxs])[0])
        n_pcs_selected.append(opt_edges_pcs)
        
    return test_rs, n_pcs_selected
