package myplayer;

import static ap25.Board.*;
import static ap25.Color.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

import ap25.*;

public class OurPlayer extends ap25.Player {
	/*
	 * 
    OurEval eval: 評価関数。
    int depthLimit: 探索の深さ制限。
    Move move: 現在選んでいる手。
    MyBoard board: 内部で保持する盤面。

	 */
  static final String MY_NAME = "OUR";
  Eval eval;
  int depthLimit;
  Move move;
  BitBoard board;

  public OurPlayer(Color color) {//デフォルト名 "MY24"、評価関数、深さ2で初期化。
    this(MY_NAME, color, new Eval(), 2);
  }

  public OurPlayer(String name, Color color, Eval eval, int depthLimit) {//名前、色、評価関数、探索深さを指定して初期化。
    super(name, color);
    this.eval = eval;
    this.depthLimit = depthLimit;
    this.board = new BitBoard();
  }

  public OurPlayer(String name, Color color, int depthLimit) {
    this(name, color, new Eval(), depthLimit);
  }

  public void setBoard(Board board) {//外部から渡された盤面を内部の MyBoard にコピー。
     for(int i=0;i<LENGTH;i++){
      this.board.set(i,board.get(i));
     }
  }

  boolean isBlack() { return getColor() == BLACK; }

  public Move think(Board board) {
    this.board = this.board.placed(board.getMove());

    if (this.board.findNoPassLegalIndexes(getColor()).size() == 0) {
      this.move = Move.ofPass(getColor());
    } else {
      var newBoard = isBlack() ? this.board.clone() : this.board.flipped();
      this.move = null;

      negaScout(newBoard, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, 0);

      this.move = this.move.colored(getColor());
    }

    this.board = this.board.placed(this.move);
    return this.move;
    /*
     * プレイヤーの思考メソッド。次の手を決定します。
	処理の流れ：
    直前の手を placed() で盤面に反映。
    合法手がなければ PASS を返す。
    自分が白番なら盤面を反転（白視点で探索）。
    maxSearch() を呼び出して最善手を探索。
    結果の手を自分の色に戻して返す。

     */
  }
//ミニマックス探索（α-β枝刈り）で最善手を評価。

  int negaScout(Board board, float alpha, float beta, int depth) {
    if (isTerminal(board, depth)) return this.eval.value(board);

    var moves = board.findLegalMoves(BLACK);
    moves = order(moves);
    boolean first = true;
    int score = Integer.MIN_VALUE;

    for (var move : moves) {
        var newBoard = board.placed(move);
        int value;

        if (first) {
            value = -negaScout(newBoard.flipped(), -beta, -alpha, depth + 1);
            first = false;
        } else {
            // Null window search
            value = -negaScout(newBoard.flipped(), -alpha - 1, -alpha, depth + 1);
            if (alpha < value && value < beta) {
                // Re-search with full window
                value = -negaScout(newBoard.flipped(), -beta, -value, depth + 1);
            }
        }

        if (value > score) {
            score = value;
            if (depth == 0) this.move = move;
        }

        alpha = Math.max(alpha, value);
        if (alpha >= beta) break;
    }

    return score;
  }

  boolean isTerminal(Board board, int depth) {//ゲーム終了または探索深さ制限に達したかを判定。
    return board.isEnd() || depth > this.depthLimit;
  }

  List<Move> order(List<Move> moves) {//手の順序をランダムにシャッフルして探索の多様性を確保。
    var shuffled = new ArrayList<Move>(moves);
    Collections.shuffle(shuffled);
    return shuffled;
  }
}
