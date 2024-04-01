# function to grab functional file and recall script and return nifti file of the areas associated with it using Ridge Regression
def extractNeuro(func_file):
    from nilearn.datasets import fetch_atlas_schaefer_2018 
    from nilearn.maskers import NiftiLabelsMasker
    # Fetch Schaefer atlas with 400 parcels and 17 Yeo networks
    n_parcels = 400
    #atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=17, resolution_mm=2)
    atlas = fetch_atlas_schaefer_2018(n_rois=400)
    # Initialize labels masker with atlas parcels
    masker = NiftiLabelsMasker(atlas.maps)
    # Fit masker to extract mean time series for parcels
    func_parcels = masker.fit_transform(func_file)
    return (masker, func_parcels)


def readText(text_file):
    # load the model
    from gensim.models import KeyedVectors
    import pandas as pd
    import numpy as np
    model = KeyedVectors.load('word2vec-google-news-300.model', mmap='r')
    
    transcript_w2v = pd.read_csv(text_file, sep ='\t')
    # Convert words to lowercase
    transcript_w2v['text'] = transcript_w2v.text.str.lower()
    # Function to extract embeddings if available
    def get_vector(word):
        if word in model.key_to_index:
            return model.get_vector(word, norm=True).astype(np.float32)
        return np.nan
    transcript_w2v['embedding'] = transcript_w2v.text.apply(get_vector)
    transcript_w2v = transcript_w2v.astype({'onset': 'float32', 'duration': 'float32'}, copy=False)
    
    return transcript_w2v

def construct_predictors(transcript_df, n_features, stim_dur, tr=1):
    import numpy as np
    # Find total number of TRs
    stim_trs = np.ceil(stim_dur / tr)

    # Add column to transcript with TR indices
    transcript_df['TR'] = transcript_df.onset.divide(tr).apply(np.floor).apply(int)

    # Compile the words within each TR
    words_per_tr = transcript_df.groupby('TR')['text'].apply(list)

    # Average the embeddings within each TR
    embeddings_per_tr = transcript_df.groupby('TR')['embedding'].mean()

    # Loop through TRs
    words_trs = []
    embeddings_trs = []
    for t in np.arange(stim_trs):
        if t in words_per_tr:
            words_trs.append(words_per_tr[t])

            # Fill in empty TRs with zero vectors
            if embeddings_per_tr[t] is not np.nan:
                embeddings_trs.append(embeddings_per_tr[t])
            else:
                embeddings_trs.append(np.zeros(n_features))
        else:
            words_trs.append([])
            embeddings_trs.append(np.zeros(n_features))

    embeddings = np.vstack(embeddings_trs)
    return embeddings


def runRidgeReg(X,Y_parcels):
    # X will be the embedding and Y will include array of timeseries per parcel
    from sklearn.preprocessing import StandardScaler
    from voxelwise_tutorials.delayer import Delayer
    from sklearn.model_selection import KFold
    from himalaya.kernel_ridge import KernelRidgeCV
    from sklearn.pipeline import make_pipeline
    import numpy as np
    # Split-half outer and inner cross-validation
    outer_cv = KFold(n_splits=2)
    inner_cv = KFold(n_splits=5)

    # Mean-center each feature (columns of predictor matrix)
    scaler = StandardScaler(with_mean=True, with_std=True)

    # Create delays at 3, 4, 5, 6, seconds (1s TR)
    delayer = Delayer(delays=[2, 3, 4, 5])

    # Ridge regression with alpha grid and nested CV
    alphas = np.logspace(1, 10, 10)
    ridge = KernelRidgeCV(alphas=alphas, cv=inner_cv)

    # Chain transfroms and estimator into pipeline
    pipeline = make_pipeline(scaler, delayer, ridge)
    Y_predicted = []
    for train, test in outer_cv.split(Y_parcels):

        # Fit pipeline with transforms and ridge estimator
        pipeline.fit(X[train],
                    Y_parcels[train])

        # Compute predicted response
        predicted = pipeline.predict(X[test])
        Y_predicted.append(predicted)

    # Restack first and second half predictions
    Y_predicted = np.vstack(Y_predicted)

    return Y_predicted


