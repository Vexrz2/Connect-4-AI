from Connect4 import Connect4
from agents import RandomAgent, DQNAgent
from ReplayBuffer import ReplayBuffer
import torch
from Tester import Tester

# Training parameters
epochs = 100000
start_epoch = 0
C = 1000
learning_rate = 0.001
batch_size = 128
env = Connect4()
MIN_Buffer = 10000 

# Training files 
File_Num = 1
path_load= f'data/params_{File_Num}.pth'
path_Save=f'data/params_{File_Num}.pth'
path_best = f'data/best_params_{File_Num}.pth'
buffer_path = f'data/buffer_{File_Num}.pth'
results_path=f'data/results_{File_Num}.pth'
random_results_path = f'data/random_results_{File_Num}.pth'
path_best_random = f'data/best_random_params_{File_Num}.pth'

def main ():
    player1 = DQNAgent(player=1, env=env,parameters_path=None)
    player2 = DQNAgent(player=-1, env=env,parameters_path=None)
    player_hat = DQNAgent(player=1, env=env,parameters_path=None)
        
    Q = player1.DQN
    player2.DQN = Q
    Q_hat = Q.copy()
    Q_hat.train = False
    player_hat.DQN = Q_hat
    

    buffer = ReplayBuffer(path=None) # buffer_path
    load = False
    
    results_file = torch.load(results_path)
    results = results_file['results'] if load else []
    avgLosses = results_file['avglosses'] if load else []
    avgLoss = avgLosses[-1] if load else 0
    loss = torch.Tensor([0])
    res = 0
    best_res = -200
    loss_count = 0
    tester = Tester(player1=player1, player2=RandomAgent(player=-1), env=env)
    random_results = torch.load(random_results_path) if load else []
    best_random = max(random_results) if load else -100
        
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000*5, gamma=0.90)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state_1 = env.get_init_state()
        state_2, action_2, after_state_2 = None, None, None
        state_2_R, action_2_R, after_state_2_R = None, None, None
        while not env.is_end_of_game(state_1):
        # Sample Environement
            # red
            action_1 = player1.get_action(state_1, epoch=epoch)
            action_1_R = action_1
            after_state_1 = env.next_state(state=state_1, col=action_1)
            after_state_1_R = after_state_1.reverse()
            reward_1, end_of_game_1 = env.reward(after_state_1)
            reward_1_R, end_of_game_1_R = -reward_1, end_of_game_1 
            if state_2: # not the first action in a game
                    buffer.push(state_2_R, action_2_R, reward_1_R, after_state_1_R, end_of_game_1_R)    # for yellow
            if end_of_game_1:
                res += reward_1
                buffer.push(state_1, action_1, reward_1, after_state_1, True) # for red
                state_1 = after_state_1
            else:
                # Yellow
                state_2 = after_state_1
                state_2_R = after_state_1_R
                action_2 = player2.get_action(state=state_2, epoch=epoch, yellow_state=state_2_R)
                action_2_R = action_2
                after_state_2 = env.next_state(state=state_2, col=action_2)
                after_state_2_R = after_state_2.reverse()
                reward_2, end_of_game_2 = env.reward(state=after_state_2)
                reward_2_R, end_of_game_2_R = -reward_2, end_of_game_2
                if end_of_game_2:
                    res += reward_2
                    buffer.push(state_2_R, action_2_R, reward_2_R, after_state_2_R, True) # for yellow
                buffer.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
                state_1 = after_state_2
            if len(buffer) < MIN_Buffer:
                continue
            
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states, actions)
            next_actions = player_hat.get_actions(next_states, dones).unsqueeze(1)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions) 

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            #scheduler.step()

            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 

        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict()) # Update target network every C epochs
        
        if (epoch+1) % 100 == 0:        # Add avg loss and res to data files every 100 epochs
            print(f'\nres= {res}')
            avgLosses.append(avgLoss)
            results.append(res)
            # if best_res < res:      
            #     best_res = res
            #     if best_res > 75:
            #         player1.save_param(path_best)
            res = 0

        if (epoch+1) % 1000 == 0:       # Test against random every 1000 epochs
            test = tester(1000)
            test_score = test[0]-test[1]
            if best_random < test_score:
                best_random = test_score
                player1.save_param(path_best_random)
            print('test',test, 'best_random:', best_random)
            random_results.append(test_score)

        if (epoch+1) % 5000 == 0:       # Save parameter data every 5000 epochs
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses}, results_path)
            torch.save(buffer, buffer_path)
            
            player1.save_param(path_Save)
            torch.save(random_results, random_results_path)
        
        if len(buffer) > MIN_Buffer:
            print (f'epoch={epoch} loss={loss:.5f} Q_values[0]={Q_values[0].item():.3f} avgloss={avgLoss:.5f}', end=" ")
            print (f'learning rate={learning_rate} path={path_Save} batch_size={batch_size}')

    torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses}, results_path)
    torch.save(buffer, buffer_path)
    
    torch.save(random_results, random_results_path)

if __name__ == '__main__':
    main()