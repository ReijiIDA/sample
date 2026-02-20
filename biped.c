#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "rcb4.h"

// --- 設定値 ---
#define SERIAL_PORT "/dev/ttyUSB0"
#define MAX_SERVO_ID 23 
#define CENTER_VALUE 7500
#define STEP_SIZE 4       // 歩行サイクル数（偶数推奨）
#define CYCLES_TO_GOAL 70
#define STEPS_PER_DEG 29.63 

// --- サーボID定義---
#define HEAD        1
#define WAIST       2
#define L_SHLDR_P   3
#define L_SHLDR_R   5
#define L_ELBW_Y    7
#define L_ELBW_P    9
#define R_SHLDR_P   4
#define R_SHLDR_R   6
#define R_ELBW_Y    8
#define R_ELBW_P    10
#define L_TIGHT_Y   11
#define L_TIGHT_R   13
#define L_TIGHT_P   15
#define L_KNEE      17
#define L_FOOT_P    19
#define L_FOOT_R    21
#define R_TIGHT_Y   12
#define R_TIGHT_R   14
#define R_TIGHT_P   16
#define R_KNEE      18
#define R_FOOT_P    20
#define R_FOOT_R    22

int deg2pulse(float degree) {
    return (int)(CENTER_VALUE + (degree * STEPS_PER_DEG));
}
 
typedef struct {
    int positions[MAX_SERVO_ID];
    int speed;
    int wait_time_ms;
} Posture;

// 関数プロトタイプ
int send_posture(rcb4_connection* conn, Posture p);
void create_biped_motion(Posture* steps, Posture* init_pose);
void init_posture_data(Posture* p);
void mirror_posture(const Posture *a, Posture *b);
void symmetrize_posture(Posture* p);

int main() {
    rcb4_connection* conn = rcb4_init(SERIAL_PORT);
    if (!conn) {
        printf("Connection failed.\n");
        return -1;
    }
    printf("Robot Connected.\n");

    Posture init_pose;
    Posture walk_cycle[STEP_SIZE]; 
    
    init_posture_data(&init_pose);
    for(int i=0; i<STEP_SIZE; i++) {
        init_posture_data(&walk_cycle[i]);
    }

    create_biped_motion(walk_cycle, &init_pose);

    printf("Moving to Initial Position...\n");
    if (send_posture(conn, init_pose) < 0) {
        printf("Init failed.\n");
        rcb4_deinit(conn);
        return -1;
    }
    sleep(1);

    printf("Walking start... (Press Ctrl+C to stop)\n");
    
    int current_step = 0;
    int sgn = 1;
    
    while(1) {
        int idx = current_step % STEP_SIZE;
        if (idx < 0) idx += STEP_SIZE;

        if (send_posture(conn, walk_cycle[idx]) < 0) {
            printf("Command error at step %d\n", idx);
            break;
        }

        current_step += sgn;

        // ゴールに達したら反転、または0に戻ったら反転
        if (current_step >= CYCLES_TO_GOAL) {
            sgn = -1;
        } else if (current_step <= 0) {
            sgn = 1;
        }
    }

    rcb4_deinit(conn);
    return 0;
}

void mirror_posture(const Posture *a, Posture *b) {
    // 速度と待機時間はそのままコピー
    b->speed = a->speed;
    b->wait_time_ms = a->wait_time_ms;

    for (int i = 2; i <= 11; i++) {
        int left_id = 2 * i - 1;
        int right_id = 2 * i;

        b->positions[left_id] = -a->positions[right_id];
        b->positions[right_id] = -a->positions[left_id];
        
    }
}
/*
void symmetrize_posture(Posture* p) {
    // 左半身(Odd ID)の値を反転して右半身(Even ID)に設定
    for (int i = 2; i <= 11; i++) {
        int left_id = 2 * i - 1;
        int right_id = 2 * i;

        p->positions[right_id] = -p->positions[left_id];
    }
}
*/
void init_posture_data(Posture* p) {
    for(int i=0; i<MAX_SERVO_ID; i++){
        p->positions[i] = 0;
    }
    p->speed = 350;
    p->wait_time_ms = 1000; // 速めの切り替え
}

int send_posture(rcb4_connection* conn, Posture p) {
    rcb4_comm* comm = rcb4_command_create(RCB4_COMM_CONST);
    rcb4_command_set_speed(comm, p.speed);
    for (int i = 1; i < MAX_SERVO_ID; i++) {
        // 0のIDは送らない
        if(p.positions[i] != 0 || i == HEAD) { // 簡易的な最適化
             rcb4_command_set_servo(comm, i, 0, deg2pulse(p.positions[i])); 
        }
    }
    int ret = rcb4_send_command(conn, comm, NULL);
    rcb4_command_delete(comm);
    rcb4_util_usleep(p.wait_time_ms * 1000);
    return ret;
}

// biped.c の create_biped_motion をこれに置き換えてください

void create_biped_motion(Posture* steps, Posture* init_pose) {

    int tilt = 9;
    init_pose->positions[L_SHLDR_P] = 30;
    init_pose->positions[L_TIGHT_P] = 85;
    init_pose->positions[L_KNEE] = 60;
    init_pose->positions[L_FOOT_P] = 10;
    init_pose->positions[R_TIGHT_R] = -tilt;
    init_pose->positions[L_TIGHT_R] = -tilt;
    init_pose->positions[R_TIGHT_P] = -90;
    init_pose->positions[R_KNEE] = -60;
    init_pose->positions[R_FOOT_P] = 5;

    for(int i=0; i<STEP_SIZE; i++) {
        steps[i] = *init_pose;
        steps[i].speed = 350;
        steps[i].wait_time_ms = 1000; 
    }

    // --- 歩行サイクル (STEP_SIZE = 4) ---
    steps[0] = *init_pose;
    steps[0].positions[L_TIGHT_R] = -tilt;
    steps[0].positions[L_KNEE] = 10;
    steps[0].positions[L_FOOT_P] = 55;
    steps[0].positions[L_FOOT_R] = 8;
    steps[0].positions[R_KNEE] = -120;
    steps[0].positions[R_FOOT_P] = 55;
    steps[0].positions[R_FOOT_R] = tilt;
    steps[0].positions[L_TIGHT_P] = 90;
    // --- Step 1: 少しニュートラルを経由（安定化） ---
    // Step 0の状態から、足の前後幅はそのままで、傾きだけ戻すイメージ
    steps[1] = steps[0];
    steps[1].positions[L_SHLDR_P] = 100;
    steps[1].positions[R_SHLDR_P] = -100;
    steps[1].positions[L_TIGHT_P] = 75;
    steps[1].positions[L_TIGHT_R] = tilt;
    steps[1].positions[R_TIGHT_R] = tilt;
    steps[1].positions[L_FOOT_R] = -tilt;
    steps[1].positions[R_FOOT_R] = -tilt;

    // --- Step 2 & 3: 右側への動作（Step 0,1 のミラー） ---
    // 左右反転関数を利用して生成
    mirror_posture(&steps[0], &steps[2]);
    mirror_posture(&steps[1], &steps[3]);
}