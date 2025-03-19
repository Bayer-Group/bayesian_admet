import numpy as np
import arviz as az
import pymc as pm

import dili_predict as dp

random_state = np.random.RandomState(674968)
dili_posterior_path = dp.path.Experiments.dili_prediction / "dili_posterior.nc"
if not dili_posterior_path.exists():
    print("Posterior for DILI model not found, running experiment...")
    dp.experiment_runner.dili_experiment(random_state)

def predict_dili(smiles, name, MTX_MP, PLD, ROS, CTX, BSEPi, DILI_majority):
    model = dp.model.binary_dili(
        MTX_MP=MTX_MP,
        PLD=PLD,
        ROS=ROS,
        CTX=CTX,
        BSEPi=BSEPi,
        missing_MTX_MP=0,
        missing_PLD=0,
        missing_ROS=0,
        missing_CTX=0,
        missing_BSEPi=0,
        p_MTX_MP=0,  # this will be ignored
        p_PLD=0,  # this will be ignored
        p_ROS=0,  # this will be ignored
        p_CTX=0,  # this will be ignored
        p_BSEPi=0,  # this will be ignored
        DILI_majority=DILI_majority,
    )

    posterior = az.from_netcdf(dp.path.Experiments.dili_prediction / "dili_posterior.nc")

    print(f"{name}: Predicting DILI for complete model")
    with model:
        pp_complete = pm.sample_posterior_predictive(
            posterior, model, var_names=["DILI_pred", "p"], random_seed=random_state
        )
        pp_complete.to_netcdf(
            dp.path.Experiments.dili_prediction / f"{name}_full_pp.nc"
        )

    pred_MTX_MP = dp.experiment_runner.predict_for_smiles("MTX_MP", smiles)
    pred_PLD = dp.experiment_runner.predict_for_smiles("PLD", smiles)
    pred_BSEPi = dp.experiment_runner.predict_for_smiles("BSEPi", smiles)
    pred_ROS = dp.experiment_runner.predict_for_smiles("ROS", smiles)
    pred_CTX = dp.experiment_runner.predict_for_smiles("CTX", smiles)

    print(f"{name}: Predicting DILI for partially imputed model")
    with model:
        pm.set_data(
            {
                "missing_MTX_MP": np.array(0),
                "missing_PLD": np.array(0),
                "missing_ROS": np.array(1),
                "missing_CTX": np.array(1),
                "missing_BSEPi": np.array(0),
                "p_MTX_MP": np.array(0), # ignored
                "p_PLD": np.array(0), # ignored
                "p_ROS": pred_ROS,
                "p_CTX": pred_CTX,
                "p_BSEPi": np.array(0), # ignored
                "DILI": np.array(1),
            }
        )
        pp_impute = pm.sample_posterior_predictive(
            posterior, model, var_names=["DILI_pred", "p"], random_seed=random_state
        )
        pp_impute.to_netcdf(
            dp.path.Experiments.dili_prediction / f"{name}_partial_impute_pp.nc"
        )

    print(f"{name}: Predicting DILI for fully imputed model")
    with model:
        pm.set_data(
            {
                "missing_MTX_MP": np.array(1),
                "missing_PLD": np.array(1),
                "missing_ROS": np.array(1),
                "missing_CTX": np.array(1),
                "missing_BSEPi": np.array(1),
                "p_MTX_MP": pred_MTX_MP,
                "p_PLD": pred_PLD,
                "p_ROS": pred_ROS,
                "p_CTX": pred_CTX,
                "p_BSEPi": pred_BSEPi,
                "DILI": np.array(1),
            }
        )
        pp_impute = pm.sample_posterior_predictive(
            posterior, model, var_names=["DILI_pred", "p"], random_seed=random_state
        )
        pp_impute.to_netcdf(
            dp.path.Experiments.dili_prediction / f"{name}_impute_pp.nc"
        )

atorvastatin = {
                    "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
                    "name": "atorvastatin",
                    "MTX_MP": np.array(1),
                    "PLD": np.array(0),
                    "ROS": np.array(1),
                    "CTX": np.array(1),
                    "BSEPi": np.array(1),
                    "DILI_majority": np.array(1)
                }

acetaminophen = {
                    "smiles": "CC(=O)Nc1ccc(O)cc1",
                    "name": "acetaminophen",
                    "MTX_MP": np.array(0),
                    "PLD": np.array(0),
                    "ROS": np.array(0),
                    "CTX": np.array(0),
                    "BSEPi": np.array(0),
                    "DILI_majority": np.array(1)
                }

perhexiline = {
                    "smiles": "C1CCC(C(CC2CCCCN2)C2CCCCC2)CC1",
                    "name": "perhexiline",
                    "MTX_MP": np.array(1),
                    "PLD": np.array(1),
                    "ROS": np.array(1),
                    "CTX": np.array(1),
                    "BSEPi": np.array(0),
                    "DILI_majority": np.array(1)
                }

chlorpromazine = {
                    "smiles": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
                    "name": "chlorpromazine",
                    "MTX_MP": np.array(1),
                    "PLD": np.array(0),
                    "ROS": np.array(1),
                    "CTX": np.array(1),
                    "BSEPi": np.array(0),
                    "DILI_majority": np.array(1)
                }


for compound in [atorvastatin, acetaminophen, perhexilene, chlorpromazine]:
    print(f"Predicting DILI for {compound['name']}")
    predict_dili(
        smiles=compound["smiles"],
        name=compound["name"],
        MTX_MP=compound["MTX_MP"],
        PLD=compound["PLD"],
        ROS=compound["ROS"],
        CTX=compound["CTX"],
        BSEPi=compound["BSEPi"],
        DILI_majority=compound["DILI_majority"]
    )