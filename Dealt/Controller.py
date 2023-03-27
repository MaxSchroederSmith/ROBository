#!python3.7
import subprocess
import shootCard
import select
import re
import datetime
import blackjack
import card_detector
import bluetooth_manager
import stepper
import opening_sequence

acceptable_inputs = ["Hit","Stand","Split"]

PIN, COUNT = 18,60

strip = GroveWS2813RgbStrip(PIN, COUNT)


class Player:
    def __init__(self):
        hand = blackjack.Hand()
        hand.set_player(self)
        self.hands = [hand]
        self.score = []
        self.playerNo = 0
        self.hands_won = 0
        self.hands_won_total = 0
        
    def getPlayerNo(self):
        return self.playerNo
    
    def setPlayerNo(self, x):
        self.playerNo = x


    
def init_game(game_no, cs):
    no_players = int(input("How many players? "))
    global players
    players = []    
    for i in range(no_players):
        player = Player()
        player.setPlayerNo((i+1))
        print(player.getPlayerNo())
        players.append(player)
        
        
    global dealer
    dealer = blackjack.Hand()
    
    print("Game "+str(game_no))
    print("")
    deal_starting_hands(cs)

def deal_starting_hands(cs):
    print("Dealer")
    dealer.reset_hand()
    give_card(dealer,False)
    give_card(dealer,True)
    print("Enter to Continue")
    input("")
    for i,player in enumerate(players):
        player.hands[0].reset_hand()
        split = False
        print("Player " + str((i+1)))
        give_card(player.hands[0],False)
        give_card(player.hands[0],False)
        player.hands[0].show_hand()
        

def play_game(cs):
    
    for player in players:
        play_hand(player.hands[0],cs)
    
    play_hand_dealer(dealer)
    
    

def play_hand(hand,cs):
    if hand.splitable():
        x = 1
    else:
        x = 0
        
    num = hand.player.getPlayerNo()
    st = 'Player ' + str(num) + ':/' + str(hand.calc_hand()) + '/' + str(x)
    bluetooth_manager.sendData(st,cs)
    
    if hand.splitable():
        print("hit, stand or split?")
        line = bluetooth_manager.recieveData(cs)
        if "Split" in str(line):
            newHand = hand.split()
            hand.player.hands.append(newHand)
            play_split(hand.player, cs)
    else:
        
        while hand.calc_hand() <= 21:
            print("hit or stand?")
            line = bluetooth_manager.recieveData(cs)
            if "Hit" in str(line):
                give_card(hand,False)
                hand.show_hand()
                num = hand.player.getPlayerNo()
                st = 'Player ' + str(num) + ':/' + str(hand.calc_hand()) + '/' + str(0)
                bluetooth_manager.sendData(st,cs)
                if hand.calc_hand() > 21:
                    break
                
            elif "Stand" in str(line):
                break
            elif "Split" in str(line):
                print("you can't split.")
    hand.player.score.append(hand.calc_hand())
    if hand.calc_hand() > 21:
        print("Player 1 went bust as they got "+str(hand.calc_hand()))
    else:
        print("Player 1 got a hand value of "+str(hand.calc_hand()))
    print("")



def play_hand_dealer(dealer):
    dealer.show_hand()
    while dealer.calc_hand() < 17:
        give_card(dealer,False)
        if dealer.calc_hand() > 21:
            print("Dealer went bust as they got "+str(dealer.calc_hand()))
        else:
            print("Dealer got a hand value of "+str(dealer.calc_hand()))

    
    
def play_split(player, cs):
    for i, hand in enumerate(player.hands):
        print("Player Hand "+str(i+1))
        give_card(hand, False)
        hand.show_hand()
        num = hand.player.getPlayerNo()
        st = 'Player ' + str(num) + ':/' + str(hand.calc_hand()) + '/' + str(0)
        bluetooth_manager.sendData(st,cs)
        print("hit or stand?")
        line = bluetooth_manager.recieveData(cs)
        while hand.calc_hand() <= 21:
            if "Hit" in str(line):
                give_card(hand,False)
                num = hand.player.getPlayerNo()
                st = 'Player ' + str(num) + ':/' + str(hand.calc_hand()) + '/' + str(x)
                bluetooth_manager.sendData(st,cs)
                hand.show_hand()
                if hand.calc_hand() > 21:
                    break
                print("hit or stand?")
                num = hand.player.getPlayerNo()
                st = 'Player ' + str(num) + ':/' + str(hand.calc_hand()) + '/' + str(x)
                bluetooth_manager.sendData(st,cs)
                line = bluetooth_manager.recieveData(cs)
            elif "Stand" in str(line):
                break
            elif "Split" in str(line):
                print("you can't split. Hit or stand?")
                line = bluetooth_manager.recieveData(cs)
        if hand.calc_hand() > 21:
            print("Player Hand "+str(i+1)+" went bust as they got "+str(hand.calc_hand()))
        else:
            print("Player Hand "+str(i+1)+" got a hand value of "+str(hand.calc_hand()))
        print("")
        
    
    
def readline():
    current_time = get_current_time()
    internal_line = p.stdout.readline()
    while not any(substring in str(internal_line) for substring in acceptable_inputs) or current_time > datetime.datetime.strptime(re.search(".. ..:..:..",str(internal_line)).group(),'%d %H:%M:%S'):
        internal_line = p.stdout.readline()
        #print(internal_line)
 

    return internal_line

def get_current_time():
    getdatetime = subprocess.Popen("adb shell date",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (getdatetimereadlines, err) = getdatetime.communicate()
    print(str(getdatetime.communicate()))
    regex = re.search("[0-9]+ ..:..:..",str(getdatetimereadlines)).group()
    internal_current_time = datetime.datetime.strptime(regex,"%d %H:%M:%S")
    return internal_current_time

def give_card(player, flip):
    player.add_card(card_detector.main())
    if not flip:
        shootCard.shoot_card_no_flip()
    else:
        shootCard.shoot_card_flip()
            


    

#from asyncio.subprocess import PIPE, STDOUT
#p = subprocess.Popen("C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe -d logcat System.out:I *:S",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# subprocess.run('C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe shell am start -n "com.example.androidapp/com.example.myapplication.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER',shell=True)
subprocess.Popen('adb shell am start -n "com.example.androidapp/com.example.blackjack2.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER',shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
p = subprocess.Popen("adb -d logcat System.out:I *:S",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
def gameLoop(game_no, cs):
    game_no = game_no
    init_game(game_no, cs)
    play_game(cs)                
    for i, player in enumerate(players):
        for j, handScore in enumerate(player.score):
            
            if handScore <=21 and (handScore > dealer.calc_hand() or dealer.calc_hand() > 21):
                print("Player "+str(i+1)+" hand "+str(j+1)+" beat the dealer")
            elif handScore <=21 and (handScore == dealer.calc_hand() or dealer.calc_hand() > 21):
                print("player "+str(i+1)+" hand "+str(j+1)+" and dealer push")
            else:
                print("Dealer beat player 1 hand "+str(j+1))
    print("")
    input("")
    return True
    
    
    
    
def main():
    client_socket = bluetooth_manager.init_bt()
    opening_sequence.rainbowCycle(strip)
    opening_sequence.allOff(strip)
    stepper.forward(4)
    stepper.backward(4)
    game_no = 1
    nextHand = gameLoop(game_no, client_socket)
    while nextHand:
        game_no += 1
        gameLoop(game_no, client_socket)
    opening_sequence.rainbowCycle(strip)
    opening_sequence.allOff(strip)
    
main()