from typing import List

from fastapi import APIRouter, FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from service.api.exceptions import UserNotFoundError, ModelNotFoundError
from service.log import app_logger

import pickle

from service.recommend import get_recomendations_ANN


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
auth_scheme = HTTPBearer(auto_error=False)


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {"description": "Return recommendations for users."},
        401: {"description": "You are not authenticated"},
        404: {"description": "The Model or User was not found"},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme)
) -> RecoResponse:
    # if not token or token.credentials != request.app.state.api_key:
    #     raise HTTPException(
    #         status_code=401,
    #         detail="Not authenticated",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )

    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    model_names = ["ANN_tree", "LightFM_model"]

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name not in model_names:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs

    if model_name == 'ANN_tree':
        reco = get_recomendations_ANN(user_id, k_recs)

    if model_name == 'LightFM_model':
        # загружен датасет с рекомендациями от модели LightFM
        # модель LightFM обучена на гиперпараметрах:
        # "learning_rate": 0.02559650065881508,
        # "no_components": 8,
        # "loss": 'warp'

        # данные параметры подобраны с помощью Optuna

        recos_df = pickle.load(open('recos.pkl', 'rb'))
        recos_df = recos_df.loc[
            recos_df['user_id'] == user_id
            ]

        reco = recos_df['item_id'].to_list()
        if len(reco) < 10:
            # Популярное (найдено, как топ просмотренных фильмов):
            # 1) group_by (в таблице interactions)
            # 2) посчитано число строк по каждому айтему
            # 3) взято топ-10 списка
            reco = [10440, 15297, 9728, 13865, 4151,
                    3734, 2657, 4880, 142, 6809]

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
