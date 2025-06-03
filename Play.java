package myplayer;

import ap25.*;
import static ap25.Color.*;

import java.util.*;
import java.util.stream.*;

public class Play {
    public static final int MAX_ITER = 100;
    
    public static void main(String[] args) {
        // プレイヤー生成
        playMatch(new MyPlayer(BLACK), new OurPlayer(WHITE), "先手: MyPlayer vs 後手: OurPlayer");
        playMatch(new OurPlayer(BLACK), new MyPlayer(WHITE), "先手: OurPlayer vs 後手: MyPlayer");
    }
    
    static void playMatch(Player first, Player second, String title) {
        int firstWin = 0;
        int secondWin = 0;
        int draw = 0;
        
        for (int i = 0; i < MAX_ITER; i++) {
            Board board = new MyBoard();
            MyGame game = new MyGame(board, first, second);
            game.play();
            
            Color winner = game.board.winner();
            if (winner == first.getColor()) firstWin++;
            else if (winner == second.getColor()) secondWin++;
            else draw++;
        }
        
        printResult(title, first, second, firstWin, secondWin, draw);
    }
    
    static void printResult(String title, Player first, Player second, 
                            int firstWin, int secondWin, int draw) {
        System.out.println("\n=== " + title + " ===");
        System.out.println(first + " 勝利: " + firstWin + "割合" + (firstWin * 100) / MAX_ITER);
        System.out.println(second + " 勝利: " + secondWin + "割合" + (secondWin * 100) / MAX_ITER);
        System.out.println("引き分け: " + draw);
    }
}