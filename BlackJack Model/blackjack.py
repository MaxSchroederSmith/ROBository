RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

class Hand:
    def __init__(self):
        self.cards = []
        self.value = 0

    def add_card(self):
        card = input("what was the card dealt? ").upper()
        while card not in RANKS:
            print("not a valid card ")
            card = input("what was the card dealt? ").upper()
        self.cards.append(card)

    def calc_hand(self):
        self.value = 0
        non_aces = [c for c in self.cards if c != 'A']
        aces = [c for c in self.cards if c == 'A']

        for card in non_aces:
            if card in 'JQK':
                self.value += 10
            else:
                self.value += int(card)

        self.value += len(aces)
        if self.value <= 11 and len(aces)>0:
            self.value += 10
            
        return self.value

    def reset_hand(self):
        self.cards = []


def hit_validation(player_no):
    hit = input("player "+str(player_no)+" would you like another card, Y or N? ").upper()
    while hit!= "Y" and hit!="N":
        print("you did not input Y or N.")
        hit = input("player "+str(player_no)+" would you like another card, Y or N? ").upper()
    return hit

while True:
    No_of_players = int(input("How many players? "))
    players_score = []
    print("")
    player = Hand()
    for i in range(No_of_players):
        player.reset_hand()
        print("Player "+str(i+1))
        player.add_card()
        player.add_card()
        while player.calc_hand() <= 21:
            hit = hit_validation(i+1)
            if hit == "Y":
                player.add_card()
            else:
                break
        players_score.append(player.calc_hand())
        if player.calc_hand() > 21:
            print("Player "+str(i+1)+" went bust as they got "+str(player.calc_hand()))
        else:
            print("Player "+str(i+1)+" got a hand value of "+str(player.calc_hand()))
        print("")

    print("Dealer")
    player.reset_hand()
    player.add_card()
    player.add_card()
    while player.calc_hand() < 17:
        player.add_card()
    if player.calc_hand() > 21:
        print("Dealer went bust as they got "+str(player.calc_hand()))
    else:
        print("Dealer got a hand value of "+str(player.calc_hand()))
    print("")
    for i in range(No_of_players):
        if players_score[i] <=21 and (players_score[i] >= player.calc_hand() or player.calc_hand() > 21):
            print("Player "+str(i+1)+" beat the dealer")
        else:
            print("Dealer beat player "+str(i+1))
    print("")

    
        
        
        
            
    
