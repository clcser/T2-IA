#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <fstream>

#ifndef ALGORITHM
#define ALGORITHM 1  // Default to Q-learning
#endif

int algorithm = ALGORITHM;
float DELTA = 1e-4;

using namespace std;
int height_grid, width_grid, action_taken, action_taken2, current_episode;
int maxA[100][100], blocked[100][100];
float maxQ[100][100], cum_reward, Qvalues[100][100][4], reward[100][100],
    finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos, y_pos, prev_x_pos, prev_y_pos,
    blockedx, blockedy;

//////////////
// Setting value for learning parameters
int action_sel = 2;   // 1 is greedy, 2 is e-greedy
int environment = 2;  // 1 is small grid, 2 is Cliff walking
//int algorithm = 1;    // 1 is Q-learning, 2 is Sarsa
int stochastic_actions = 1;// 0 is deterministic actions, 1 for stochastic actions
int num_episodes = 2500;   // total learning episodes
float learn_rate = 0.1;    // how much the agent weights each new sample
float disc_factor = 0.9;  // how much the agent weights future rewards
float exp_rate = 0.99;      // how much the agent explores
int max_steps = 1000;
///////////////

ofstream reward_output;

void Initialize_environment() {
    if(environment == 1) {
        height_grid = 3;
        width_grid = 4;
        goalx = 3;
        goaly = 2;
        init_x_pos = 0;
        init_y_pos = 0;
    }

    if(environment == 2) {
        height_grid = 4;
        width_grid = 12;
        goalx = 11;
        goaly = 0;
        init_x_pos = 0;
        init_y_pos = 0;
    }

    for(int i = 0; i < width_grid; i++) {
        for(int j = 0; j < height_grid; j++) {
            if(environment == 1) {
                reward[i][j] = -0.04;
                blocked[i][j] = 0;
            }

            if(environment == 2) {
                reward[i][j] = -1;
                blocked[i][j] = 0;
            }

            for(int k = 0; k < 4; k++) {
                Qvalues[i][j][k] = (rand() % 10);
                cout << "Initial Q value of cell [" << i << ", " << j << "] action " << k << " = " << Qvalues[i][j][k] << "\n";
            }
        }
    }

    if(environment == 1) {
        reward[goalx][goaly] = 100;
        reward[goalx][(goaly-1)] = -100;
        blocked[1][1] = 1;
    }
    
    if(environment == 2) {
        reward[goalx][goaly] = 1;

        for(int h = 1; h < goalx; h++) {
            reward[h][0] = -100;
        }
    }
}

int action_selection() {
    pair<float, int> highest = {Qvalues[x_pos][y_pos][0], 0};
    for(int i = 1; i < 4; ++i) {
        highest = max(highest, {Qvalues[x_pos][y_pos][i], i});
    }

    if(action_sel == 2) { // epsilon-greedy
        float ran = (rand()%100)/100.0;
        if(ran < 1-exp_rate) {
            return highest.second;
        }
        else {
            cout << exp_rate << "\n";
            exp_rate -= DELTA;
            return rand()%4;
        } 
    }
    else { // greedy
        return highest.second; 
    }
}

void move(int action) {
    prev_x_pos = x_pos;
    prev_y_pos = y_pos;

    if(stochastic_actions) {
        int choice = rand()%10;
        if(choice == 0) { // Moverse a la derecha
            cout << ">\n";
            action = (action+1)%4;
        } 
        if(choice == 1) { // Moverse a la izquierda
            cout << "<\n";
            action = (action - 1 + 4)%4;
        }
    }

    if(action == 0) { // Up
        if((y_pos < (height_grid-1)) and !blocked[x_pos][y_pos+1]) {
            y_pos++;
        }
    }
    if(action == 1) { // Right
        if((x_pos < (width_grid-1)) and !blocked[x_pos+1][y_pos]) {
            x_pos++;
        }
    }
    if(action == 2) { // Down
        if(y_pos > 0 and !blocked[x_pos][y_pos-1]) {
            y_pos--;
        }
    }
    if(action == 3) { // Left
        if((x_pos > 0) and !blocked[x_pos-1][y_pos]) {
            x_pos--;
        }
    }
}

float get_maxQ(int x, int y) {
    float maxq = Qvalues[x][y][0];
    for(int i = 1; i < 4; ++i) {
        maxq = max(maxq, Qvalues[x][y][i]);
    }
    return maxq;
}

void update_q_prev_state() {
    float rw = reward[x_pos][y_pos];
    float term = disc_factor * get_maxQ(x_pos, y_pos);
    float Qval = Qvalues[prev_x_pos][prev_y_pos][action_taken];
    Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate*(rw + term - Qval);
}

void update_q_prev_state_sarsa() {
    float rw = reward[x_pos][y_pos];
    float term = disc_factor*Qvalues[x_pos][y_pos][action_taken2];
    float Qval = Qvalues[prev_x_pos][prev_y_pos][action_taken];
    Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate*(rw + term - Qval);

}

void Qlearning() {
    action_taken = action_selection();
    move(action_taken);
    
    update_q_prev_state();

    cum_reward += reward[x_pos][y_pos];  
    // Add the reward obtained by the agent to the cummulative reward of the
    // agent in the current episode
}

void Sarsa() {
    move(action_taken);
    
    action_taken2 = action_selection();
    update_q_prev_state_sarsa();
    action_taken = action_taken2;

    cum_reward += reward[x_pos][y_pos];  
    // Add the reward obtained by the agent to the cummulative reward of the
    // agent in the current episode
}

void Multi_print_grid() {
    int x, y;

    for (y = (height_grid - 1); y >= 0; --y) {
        for (x = 0; x < width_grid; ++x) {
            if (blocked[x][y] == 1) {
                cout <<" \033[42m# \033[0m";

            } else {
                if ((x_pos == x) && (y_pos == y)) {
                    cout << " \033[44m" << reward[x][y] << "\033[0m";

                } else {
                    cout << " \033[31m" << reward[x][y] << "\033[0m";
                }
            }
        }
        printf("\n");
    }
}


int main(int argc, char* argv[]) {
    srand(time(NULL));

    reward_output.open("Rewards.txt");
    
    Initialize_environment();
    
    Multi_print_grid();

    for(int i = 0; i < num_episodes; i++) {
        reward_output << "\n\n Episode " << i;
        cout << "\n Episode " << i << "\n";
        current_episode = i;
        x_pos = init_x_pos;
        y_pos = init_y_pos;
        cum_reward = 0;

        int steps = 0;
        if(algorithm == 2) {
            action_taken = action_selection();
        }
        while(!(((x_pos == goalx) and (y_pos == goaly)) or 
                ((environment == 1) and (x_pos == goalx) and (y_pos == (goaly - 1))) or
                ((environment == 2) and (x_pos > 0) and (x_pos < goalx) and (y_pos == 0))
             ) and steps < max_steps) {
            //cerr << x_pos << " " << y_pos << "\n";
            if(algorithm == 1) {
                Qlearning();
            }
            if(algorithm == 2) {
                Sarsa();
            }
            steps++;
        }
        cout << "total steps: "<< steps << "\n";

        finalrw[i] = cum_reward;
        reward_output << " Total reward obtained: " << finalrw[i] << "\n";
    }
    cout << algorithm << "\n";
    reward_output.close();

    return 0;
}