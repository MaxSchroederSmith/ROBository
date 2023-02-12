import subprocess
import select
import re
import datetime
import blackjack
acceptable_inputs = ["Hit","Stand","Split"]
#from asyncio.subprocess import PIPE, STDOUT
#p = subprocess.Popen("C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe -d logcat System.out:I *:S",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# subprocess.run('C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe shell am start -n "com.example.androidapp/com.example.myapplication.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER',shell=True)

subprocess.run('C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe shell am start -n "com.example.androidapp/com.example.myapplication.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER',shell=True)
p = subprocess.Popen("C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe -d logcat System.out:I *:S",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
getdatetime = subprocess.Popen("C:\\Users\\Max\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe shell date",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
current_time = datetime.datetime.strptime(re.search("..:..:..",str(getdatetime.stdout.readlines())).group(),'%H:%M:%S').time()

print(current_time)

p.stdout.readline()
p.stdout.readline()
p.stdout.readline()

player = blackjack.Hand()
No_of_players = 1#int(input("How many players? "))

line = p.stdout.readline()
while current_time > datetime.datetime.strptime(re.search("..:..:..",str(line)).group(),'%H:%M:%S').time():
    line = p.stdout.readline()

while True:
    players_score = []
    player.reset_hand()
    print("Player 1")
    player.add_card()
    player.add_card()
    player.show_hand()
    print("hit or stand?")
    line = p.stdout.readline()
    if any(substring in str(line) for substring in acceptable_inputs):
        while player.calc_hand() <= 21:
            if "Hit" in str(line):
                player.add_card()
                player.show_hand()
                if player.calc_hand() > 21:
                    break
                print("hit or stand?")
                line = p.stdout.readline()
            elif any(substring in str(line) for substring in acceptable_inputs):
                break
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
    for i in range(No_of_players):
        if players_score[0] <=21 and (players_score[0] >= player.calc_hand() or player.calc_hand() > 21):
            print("Player 1 beat the dealer")
        else:
            print("Dealer beat player 1")
    print("")

    
    
