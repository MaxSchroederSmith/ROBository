from blackjack import Hand
from pytest import MonkeyPatch

def test_add_card(monkeypatch: MonkeyPatch, capfd):
    inputs = ["A", "1", "A"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    hand.add_card()
    assert hand.cards == ["A"]
    hand.add_card()
    output = capfd.readouterr()
    assert output.out == "not a valid card \n"
    assert hand.cards == ["A","A"]
    
def test_split(monkeypatch: MonkeyPatch):
    inputs = ["A","A"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    hand.add_card()
    hand.add_card()
    hand.split()
    assert hand.cards == ["A"]
    assert hand.cards2 == ["A"]

def test_split_same_value(monkeypatch: MonkeyPatch):
    inputs = ["10","J"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    hand.add_card()
    hand.add_card()
    assert hand.splitable() == False

def test_splitable_true(monkeypatch: MonkeyPatch):
    inputs = ["6","6"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    hand.add_card()
    hand.add_card()
    assert hand.splitable() == True

def test_splitable_false(monkeypatch: MonkeyPatch):
    inputs = ["6","9"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    hand.add_card()
    hand.add_card()
    assert hand.splitable() == False

def test_calc_hand(monkeypatch: MonkeyPatch):
    inputs = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    while inputs != []:
        hand.add_card()
    assert hand.calc_hand() == 86

def test_calc_blackjack(monkeypatch: MonkeyPatch):
    inputs = ["A","J"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    while inputs != []:
        hand.add_card()
    assert hand.calc_hand() == 21

def test_reset_hand(monkeypatch: MonkeyPatch):
    inputs = ["A","A","2"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    hand.add_card()
    hand.add_card()
    hand.split()
    hand.add_card()
    assert hand.cards == ["A","2"]
    assert hand.cards2 == ["A"]
    hand.reset_hand()
    assert hand.cards == [] and hand.cards2 == []

def test_show_hand(monkeypatch: MonkeyPatch, capfd):
    inputs = ["6","9","6"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    hand = Hand()
    while inputs != []:
        hand.add_card()
    hand.show_hand()
    output = capfd.readouterr()
    assert output.out == "Your hand is 6 9 6 \n"

def test_hand_init():
    hand = Hand()
    assert hand.cards == []
    assert hand.value == 0
    assert hand.cards2 == []
