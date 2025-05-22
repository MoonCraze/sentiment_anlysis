import os
import json
from experta import *

# — Facts —
class CoinFlowFact(Fact):
    """Aggregated net-flow for a coin."""
    name = Field(str, mandatory=True)
    net_flow = Field(float, mandatory=True)

class SentimentFact(Fact):
    """Sentiment score between –1 and +1."""
    name = Field(str, mandatory=True)
    score = Field(float, mandatory=True)

# — Expert engine —
class TradingAdvisor(KnowledgeEngine):
    @DefFacts()
    def init(self):
        yield Fact(action="analyze")

    @Rule(
        Fact(action='analyze'),
        CoinFlowFact(net_flow=P(lambda x: x > 1_000_000), name=MATCH.coin),
        SentimentFact(score=P(lambda s: s > 0.7), name=MATCH.coin)
    )
    def strong_buy(self, coin):
        self.declare(Fact(recommend=coin, action='BUY',
                          reason="High net-flow & very positive sentiment"))

    @Rule(
        Fact(action='analyze'),
        CoinFlowFact(net_flow=P(lambda x: x < -1_000_000), name=MATCH.coin),
        SentimentFact(score=P(lambda s: s < -0.7), name=MATCH.coin)
    )
    def strong_sell(self, coin):
        self.declare(Fact(recommend=coin, action='SELL',
                          reason="High outflow & very negative sentiment"))

    @Rule(
        Fact(action='analyze'),
        CoinFlowFact(net_flow=P(lambda x: abs(x) < 500_000), name=MATCH.coin)
    )
    def moderate_hold(self, coin):
        self.declare(Fact(recommend=coin, action='HOLD',
                          reason="Net-flow within ±500K"))

    @Rule(Fact(recommend=MATCH.coin, action=MATCH.act, reason=MATCH.r))
    def collect(self, coin, act, r):
        print(f"{coin}: {act} — {r}")
        # also write to JSON at the end if you like

if __name__ == '__main__':
    # 1. Load data files
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, '..', 'data')

    with open(os.path.join(data_dir, 'coin_flow_data.json')) as f:
        flows = json.load(f)['aggregated_flows']
    with open(os.path.join(data_dir, 'sentiment_data.json')) as f:
        sents = json.load(f)  # e.g. {"BTC":0.8, "ETH":0.2, ...}

    # 2. Run engine
    engine = TradingAdvisor()
    engine.reset()
    for coin, net in flows.items():
        engine.declare(CoinFlowFact(name=coin, net_flow=net))
    for coin, score in sents.items():
        engine.declare(SentimentFact(name=coin, score=score))
    engine.run()

    # 3. (Optional) capture printed output into output/recommendations.json
