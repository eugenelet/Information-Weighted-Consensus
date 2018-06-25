classdef CKF_
  properties
    x  % state estimate
    V  % information matrix
    v  % information vector
    H  % observation matrix
    phi  % state transition matrix
    Q  % process covariance
    W  % prior information matrix
  end
  methods
    function obj = CKF_(x, W, H, phi, Q)
      if nargin > 0
        obj.x = x;
        obj.W = W;
        obj.H = H;
        obj.phi = phi;
        obj.Q = Q;
      end
    end
    function obj = compute_information(obj, B, z)
      if nargin > 0
        obj.V = obj.W + obj.H'*B*obj.H;
        obj.v = obj.W*obj.x + obj.H'*B*z;
      end
    end


    function [obj, W, x] = compute_posterior_state_and_info(obj)
      obj.x = inv(obj.V) * obj.v;
      obj.W = obj.V;
      x = obj.x;
      W = obj.W;
    end

    function obj = predict_next_step(obj)
      obj.W = inv(obj.phi * inv(obj.W) * obj.phi' + obj.Q);
      obj.x = obj.phi * obj.x;
      % Invert velocity at boundaries
      if(obj.x(1)<=0)
        obj.x(3) = abs(obj.x(3));
      end
      if(obj.x(1)>=500)
        obj.x(3) = -abs(obj.x(3));
      end
      if(obj.x(2)<=0)
        obj.x(4) = abs(obj.x(4));
      end
      if (obj.x(2)>=500)
        obj.x(4) = -abs(obj.x(4));
      end
    end
  end
end
