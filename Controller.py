import subprocess
import select
import re
import datetime
import blackjack
acceptable_inputs = ["Hit","Stand","Split"]

def readline():
    current_time = get_current_time()
    internal_line = p.stdout.readline()
    while not any(substring in str(internal_line) for substring in acceptable_inputs) or current_time > datetime.datetime.strptime(re.search("..-.. ..:..:..",str(internal_line)).group(),'%m-%d %H:%M:%S'):
        internal_line = p.stdout.readline()
    return internal_line

def get_current_time():
    getdatetime = subprocess.Popen("adb.exe shell date",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    getdatetimereadlines = getdatetime.stdout.readlines()
    print(getdatetime.stderr)
    regex = re.search("...  [0-9] ..:..:..",str(getdatetimereadlines)).group()
    internal_current_time = datetime.datetime.strptime(regex,'%b  %d %H:%M:%S')
    return internal_current_time

#from asyncio.subprocess import PIPE, STDOUT
#p = subprocess.Popen("C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe -d logcat System.out:I *:S",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# subprocess.run('C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe shell am start -n "com.example.androidapp/com.example.myapplication.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER',shell=True)
subprocess.run('adb.exe shell am start -n "com.example.androidapp/com.example.blackjack2.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER',shell=True)
p = subprocess.Popen("adb.exe -d logcat System.out:I *:S",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def main():

    player = blackjack.Hand()
    No_of_players = 1#int(input("How many players? "))

    game_no = 1

    print("Game "+str(game_no))
    print("")
    players_score = []
    player.reset_hand()
    split = False
    print("Player 1")
    player.add_card()
    player.add_card()
    player.show_hand()
    if player.splitable():
        print("hit, stand or split?")
        line = readline()
        if "Split" in str(line):
            player.split()
            split = True
    else:
        print("hit or stand?")
        line =readline()
    if split:
        for i in range(2):
            print("Player 1 Hand "+str(i+1))
            player.add_card()
            player.show_hand()
            print("hit or stand?")
            line = readline()
            while player.calc_hand() <= 21:
                if "Hit" in str(line):
                    player.add_card()
                    player.show_hand()
                    if player.calc_hand() > 21:
                        break
                    print("hit or stand?")
                    line = readline()
                elif "Stand" in str(line):
                    break
                elif "Split" in str(line):
                    print("you can't split. Hit or stand?")
                    line = readline()
            players_score.append(player.calc_hand())
            if player.calc_hand() > 21:
                print("Player 1 Hand "+str(i+1)+" went bust as they got "+str(player.calc_hand()))
            else:
                print("Player 1 Hand "+str(i+1)+" got a hand value of "+str(player.calc_hand()))
            print("")
            player.cards = player.cards2
                
    else:
        while player.calc_hand() <= 21:
            if "Hit" in str(line):
                player.add_card()
                player.show_hand()
                if player.calc_hand() > 21:
                    break
                print("hit or stand?")
                line = readline()
            elif "Stand" in str(line):
                break
            elif "Split" in str(line):
                print("you can't split. Hit or stand?")
                line = readline()
                
        players_score.append(player.calc_hand())
        if player.calc_hand() > 21:
            print("Player 1 went bust as they got "+str(player.calc_hand()))
        else:
            print("Player 1 got a hand value of "+str(player.calc_hand()))
        print("")

    if not(all(score>21 for score in players_score)):
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
    if split:
        for i in range(No_of_players):
            for j in range(2):
                if players_score[j] <=21 and (players_score[j] > player.calc_hand() or player.calc_hand() > 21):
                    print("Player 1 hand "+str(j+1)+" beat the dealer")
                elif players_score[j] <=21 and (players_score[j] == player.calc_hand() or player.calc_hand()>21):
                    print("player 1 hand "+str(j+1)+" and dealer push")
                else:
                    print("Dealer beat player 1 hand "+str(j+1))
    else:
        for i in range(No_of_players):
            if players_score[0] <=21 and (players_score[0] > player.calc_hand() or player.calc_hand() > 21):
                print("Player 1 beat the dealer")
            elif players_score[0] <=21 and (players_score[0] == player.calc_hand() or player.calc_hand()>21):
                print("player 1 and dealer push")
            else:
                print("Dealer beat player 1")
    print("")
    game_no+=1

main()
    
