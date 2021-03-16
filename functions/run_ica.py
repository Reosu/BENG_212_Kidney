import pandas as pd
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr, spearmanr


def run_ica(norm_data, iterations=3, n_comp=300, threshold=.9):
    # Initialize the M and A with one run of ICA
    ica_transformer = FastICA(n_components=n_comp).fit(norm_data.transpose())
    M = pd.DataFrame(ica_transformer.transform(norm_data.transpose()))
    A = pd.DataFrame(ica_transformer.mixing_)
    for i in range(0, iterations):
        print("Running iteration: " + str(i))
        ica_transformer = FastICA(n_components=n_comp).fit(
            norm_data.transpose())
        temp_M = pd.DataFrame(ica_transformer.transform(norm_data.transpose()))
        temp_A = pd.DataFrame(ica_transformer.mixing_)
        M, A = _cluster_comp(M, temp_M, A, temp_A, threshold)
    return M, A


def _cluster_comp(final_M, new_M, final_A, new_A, threshold):
    metrics = pd.DataFrame(index=range(0, len(final_M.columns)),
                           columns=range(0, len(new_M.columns)))
    for i in range(0, len(final_M.columns)):
        for j in range(0, len(new_M.columns)):
            metrics.loc[i][j] = abs(pearsonr(final_M[i], new_M[j])[0])
    metrics = metrics.fillna(0)
    M = pd.DataFrame(index=final_M.index)
    A = pd.DataFrame(index=final_A.index)
    for i, item in metrics.iteritems():
        for j in item.index:
            if item[j] == max(item) and max(item) > threshold:
                M1_abs_max = float(final_M[j].loc[final_M[j].abs().nlargest(
                    1).index])
                M2_abs_max = float(new_M[i].loc[new_M[i].abs().nlargest(
                    1).index])
                A1_abs_max = float(final_A[j].loc[final_A[j].abs().nlargest(
                    1).index])
                A2_abs_max = float(new_A[i].loc[new_A[i].abs().nlargest(
                    1).index])
                if M1_abs_max < 0:
                    M1_corrected = -final_M[j]
                else:
                    M1_corrected = final_M[j]
                if M2_abs_max < 0:
                    M2_corrected = -new_M[j]
                else:
                    M2_corrected = new_M[j]
                if A1_abs_max < 0:
                    A1_corrected = -final_A[i]
                else:
                    A1_corrected = final_A[i]
                if A2_abs_max < 0:
                    A2_corrected = -new_A[i]
                else:
                    A2_corrected = new_A[i]
                M[str(i) + "_" + str(j)] = (M1_corrected + M2_corrected) / 2
                A[str(i) + "_" + str(j)] = (A1_corrected + A2_corrected) / 2
    M.columns = range(len(M.columns))
    A.columns = range(len(A.columns))
    print("Number of Components: "+ M.shape[0])
    return M, A
