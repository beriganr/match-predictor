from matchpredictor.matchresults.result import Fixture, Outcome
from matchpredictor.predictors.predictor import Prediction, Predictor


class AlphabeticalPredictor(Predictor):
    def predict(self, fixture: Fixture) -> Prediction:

        if fixture.away_team.name < fixture.home_team.name:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.HOME)
        #return Prediction(outcome=Outcome.HOME)
