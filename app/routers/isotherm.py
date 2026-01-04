"""Isotherm fitting API endpoints"""
from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np

from app.models.schemas import (
    IsothermFitRequest,
    IsothermFitResult,
    IsothermModelType,
)
from app.services.isotherm_service import fit_isotherm, fit_best_isotherm

router = APIRouter()


@router.post("/isotherm/fit", response_model=List[IsothermFitResult])
async def fit_isotherm_models(request: IsothermFitRequest) -> List[IsothermFitResult]:
    """
    Fit isotherm models to experimental data.

    Fits the specified models to the provided pressure-loading data
    and returns fitted parameters with goodness of fit metrics.

    Args:
        request: IsothermFitRequest with data points and models to fit

    Returns:
        List of IsothermFitResult with fitted parameters
    """
    try:
        # Extract data
        pressures = np.array([p.pressure for p in request.data_points])
        loadings = np.array([p.loading for p in request.data_points])

        if len(pressures) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 data points required for fitting",
            )

        results = []
        for model in request.models_to_fit:
            model_name = model.value
            if model_name == "dubinin_radushkevich":
                model_name = "dr"
            elif model_name == "dubinin_astakhov":
                model_name = "da"

            fit_result = fit_isotherm(pressures, loadings, model_name)

            results.append(
                IsothermFitResult(
                    model=model,
                    parameters=fit_result.params,
                    r2=fit_result.r2,
                    rmse=fit_result.rmse,
                )
            )

        # Sort by R² (best first)
        results.sort(key=lambda x: x.r2, reverse=True)

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fitting error: {str(e)}")


@router.post("/isotherm/predict")
async def predict_loading(
    pressure: float,
    model: IsothermModelType,
    parameters: dict,
):
    """
    Predict loading at a given pressure using isotherm model.

    Args:
        pressure: Pressure or concentration
        model: Isotherm model type
        parameters: Model parameters

    Returns:
        Predicted loading
    """
    from app.services.isotherm_service import predict_loading, IsothermParams

    try:
        model_name = model.value
        if model_name == "dubinin_radushkevich":
            model_name = "dr"
        elif model_name == "dubinin_astakhov":
            model_name = "da"

        params = IsothermParams(model=model_name, params=parameters)
        loading = predict_loading(pressure, params)

        return {"pressure": pressure, "loading": loading, "model": model.value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/isotherm/models")
async def list_available_models():
    """
    List available isotherm models with descriptions.

    Returns:
        Dictionary of models with descriptions and parameters
    """
    return {
        "models": {
            "langmuir": {
                "name": "Langmuir",
                "equation": "q = q_max * K * p / (1 + K * p)",
                "parameters": ["q_max", "K"],
                "description": "Monolayer adsorption, homogeneous surface",
            },
            "freundlich": {
                "name": "Freundlich",
                "equation": "q = K * p^(1/n)",
                "parameters": ["K", "n"],
                "description": "Empirical, heterogeneous surface",
            },
            "sips": {
                "name": "Sips (Langmuir-Freundlich)",
                "equation": "q = q_max * (K*p)^(1/n) / (1 + (K*p)^(1/n))",
                "parameters": ["q_max", "K", "n"],
                "description": "Combines Langmuir and Freundlich",
            },
            "toth": {
                "name": "Toth",
                "equation": "q = q_max * K * p / (1 + (K*p)^t)^(1/t)",
                "parameters": ["q_max", "K", "t"],
                "description": "Asymmetric quasi-Gaussian distribution",
            },
            "dr": {
                "name": "Dubinin-Radushkevich",
                "equation": "W = W_0 * exp(-(RT/E * ln(p_sat/p))²)",
                "parameters": ["W_0", "E", "p_sat"],
                "description": "Micropore volume filling theory",
            },
            "da": {
                "name": "Dubinin-Astakhov",
                "equation": "W = W_0 * exp(-(RT/E * ln(p_sat/p))^n)",
                "parameters": ["W_0", "E", "n", "p_sat"],
                "description": "Generalized DR model",
            },
        }
    }
