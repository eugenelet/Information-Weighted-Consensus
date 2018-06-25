%%%%%%%%%%%%%%%%%%%%%%%
% Fixed Parameters    %
%%%%%%%%%%%%%%%%%%%%%%%
clear;
phi = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];  % State Transition Matrix
Q = [10 0 0 0; 0 10 0 0; 0 0 1 0; 0 0 0 1];  % Process Covariance
H = [1 0 0 0; 0 1 0 0];  % Observation Matrix
A = [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0];  % Communication Adjacency Matrix
B = 0.02 * eye(2);  % Measurement Information
nu = inv(B);  % ground_truth. Noise Covariance
deg = 2;  % Network degree
K = 100;  % Total Consensus Iteration
eps = 0.65 / deg;  % Consensus rate parameter
sensor_n = 5;  % Number of sensors
time_steps = 40;  % Number of time steps for target
simulations = 15;
% Field of View
FOV = cat(3, [13 13; 357 12; 200 350; 13 13], [100 130; 400 130; 300 490; 100 130], [250 490; 490 492; 400 200; 250 490], [10 490; 12 170; 270 300; 10 490], [200 150; 490 14; 440 290; 200 150]);


error = 0;



%%%%%%%%%%%%%%%%%%%%%%%
% Initital Parameters %
%%%%%%%%%%%%%%%%%%%%%%%
% Initial location
x = [250;250;0;0];
% Set initial velocity
angle = 2*pi*rand(1);
speed = randi([10 20]);
x(3) = speed * cos(angle);
x(4) = speed * sin(angle);


P = [100 0 0 0; 0 100 0 0; 0 0 10 0; 0 0 0 10];  % Prior Covariance
W = inv(P);  % Prior information matrix
x_ICF = x + mvnrnd(zeros(4,1), P)';  % Prior estimate

node = [];  % nodes for ICF
for i = 1:5
  node = [node, ICF_node(x_ICF, W, H, eps, K, phi, Q)];
end
% Centralized Kalman Filter
CKF = CKF_(x_ICF, W, H, phi, Q);


ICF_error = [];
CKF_error = [];
for sim = 1:simulations
  CKF_error(sim) = 0;
  ICF_error(sim) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%
  % Start Tracking!!    %
  %%%%%%%%%%%%%%%%%%%%%%%
  obs = [];
  for i = 1:time_steps
      % Update state of target
      x = phi*x + mvnrnd(zeros(4,1), Q)';
      % Ground truth track of target
      ground_truth = H*x;
      % Observed track of target
      obs = [obs, H*x + mvnrnd(zeros(2,1), nu)'];

      % Invert velocity at boundaries
      if(x(1)<=0)
        x(3) = abs(x(3));
      end
      if(x(1)>=500)
        x(3) = -abs(x(3));
      end
      if(x(2)<=0)
        x(4) = abs(x(4));
      end
      if (x(2)>=500)
        x(4) = -abs(x(4));
      end

      % CKF
      CKF = CKF.compute_information(B, obs(:,end));
      [CKF, CKF_est_W, CKF_est_x] = CKF.compute_posterior_state_and_info();
      CKF = CKF.predict_next_step();
      CKF_est_z = H*CKF_est_x;

      % CKF error
      CKF_error(sim) = CKF_error(sim) + sum(sqrt((CKF_est_z - ground_truth).^2));

      % Compute information matrix and vector
      for j = 1:sensor_n
        [obs_in, obs_on] = inpolygon(obs(1,end),obs(2,end),FOV(:,1,j),FOV(:,2,j));
        obs_in_on = obs_in | obs_on;
        if obs_in_on  % observed by sensor
          node(j) = node(j).compute_information(B, obs(:,end));
        else
          node(j) = node(j).compute_information(0*B, obs(:,end));
        end
      end


      % Perform average consensus
      for k = 1:K
        for j = 1:sensor_n
          node(j).adj_V = {};
          node(j).adj_v = {};
          for adj = 1:sensor_n
            if A(j,adj)
              node(j).adj_V{size(node(j).adj_V, 2) + 1} = node(adj).V;
              node(j).adj_v{size(node(j).adj_v, 2) + 1} = node(adj).v;
            end
          end
        end
        for j = 1:sensor_n
          node(j) = node(j).average_consensus();
        end
      end


      error = 0;
      for j = 1:sensor_n
        % Compute posterior state estimate and information matrix for time t
        [node(j), est_W, est_x] = node(j).compute_posterior_state_and_info();
        % Prediction
        node(j) = node(j).predict_next_step();
        est_z = H*est_x;
        error = sum(sqrt((est_z - ground_truth).^2));
        ICF_error(sim) = ICF_error(sim) + error / 5;
      end
  end
end

ICF_error = ICF_error / time_steps;
CKF_error = CKF_error / time_steps;
x_axis = linspace(1,size(ICF_error,2),size(ICF_error,2));
plot(x_axis, ICF_error, x_axis, CKF_error);
legend('ICF', 'CKF')
title('Error vs Simulation Runs')
xlabel('Independent Simulation Runs')
ylabel('Mean Error')
