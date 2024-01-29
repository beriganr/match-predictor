from typing import List, Tuple, Optional, cast

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.linear_model import PoissonRegressor # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction

# Based on the the "How Our Club Soccer Predictions Work" article, I was inspired to implement
# a Poisson regression. The logic for processing the data remains largely the same as that
# implemented in the linear regression model. The primary difference is the use of a model
# fit to each home_goals and away_goals. The predictor produced an accuracy of 0.405263 for the
# Barclays Premier League 2021, outperforming only the alphabetical predictor.

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction

class PoissonRegressionPredictor(Predictor):
    def __init__(self, home_goals_model: PoissonRegressor, away_goals_model: PoissonRegressor, team_encoding: OneHotEncoder) -> None:
        self.home_goals_model = home_goals_model
        self.away_goals_model = away_goals_model
        self.team_encoding = team_encoding

    def predict(self, fixture: Fixture) -> Prediction:
        encoded_home_name = self.__encode_team(fixture.home_team)
        encoded_away_name = self.__encode_team(fixture.away_team)

        if encoded_home_name is None or encoded_away_name is None:
            return Prediction(outcome=Outcome.DRAW)

        x: NDArray[np.float64] = np.concatenate([encoded_home_name, encoded_away_name], axis=1)
        home_goals_pred = self.home_goals_model.predict(x)
        away_goals_pred = self.away_goals_model.predict(x)

        if home_goals_pred > away_goals_pred:
            return Prediction(outcome=Outcome.HOME)
        elif home_goals_pred < away_goals_pred:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.DRAW)

    def __encode_team(self, team: Team) -> Optional[NDArray[np.float64]]:
        try:
            result: NDArray[np.float64] = self.team_encoding.transform(np.array(team.name).reshape(-1, 1))
            return result
        except ValueError:
            return None

def build_poisson_model(results: List[Result]) -> Tuple[PoissonRegressor, PoissonRegressor, OneHotEncoder]:
    home_names = np.array([r.fixture.home_team.name for r in results])
    away_names = np.array([r.fixture.away_team.name for r in results])
    home_goals = np.array([r.home_goals for r in results])
    away_goals = np.array([r.away_goals for r in results])

    team_names = np.concatenate([home_names, away_names]).reshape(-1, 1)
    team_encoding = OneHotEncoder(sparse=False).fit(team_names)

    encoded_home_names = team_encoding.transform(home_names.reshape(-1, 1))
    encoded_away_names = team_encoding.transform(away_names.reshape(-1, 1))
    x: NDArray[np.float64] = np.concatenate([encoded_home_names, encoded_away_names], axis=1)

    home_goals_model = PoissonRegressor().fit(x, home_goals)
    away_goals_model = PoissonRegressor().fit(x, away_goals)

    return home_goals_model, away_goals_model, team_encoding

def train_poisson_predictor(results: List[Result]) -> Predictor:
    home_goals_model, away_goals_model, team_encoding = build_poisson_model(results)
    return PoissonRegressionPredictor(home_goals_model, away_goals_model, team_encoding)
