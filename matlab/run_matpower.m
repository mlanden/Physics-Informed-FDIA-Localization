define_constants;

number = "14";
mpc = loadcase('case' + number);
%[ybus, yf, yt] = makeYbus(mpc);
%writematrix(full(ybus), "14_line_1_out_ybus.csv")
%line_r = mpc.branch(:, BR_R);
%line_x = mpc.branch(:, BR_X);
%limits = ["PMIN" "PMAX" "QMIN" "QMAX"];
%for i=1:length(mpc.bus)
%    gen_idx = find(mpc.gen(:, GEN_BUS) == i);
%    if ~isempty(gen_idx)
%        info = [mpc.gen(gen_idx, PMIN) mpc.gen(gen_idx, PMAX) mpc.gen(gen_idx, QMIN) mpc.gen(gen_idx, QMAX)]; 
%    else
%        info = [0 0 0 0];
%    end
%    limits = [limits; info];
%end
%writematrix(limits, "14_limits.csv");

%branch_data = ["r" "x" ];
%branch_data = [branch_data; line_r line_x];
%writematrix(branch_data, "14_branch.csv");
extract_loads(mpc, number) 
load_mw = readtable(number + '_MW.csv', 'NumHeaderLines', 1);
load_mvar = readtable(number + '_Mvar.csv', 'NumHeaderLines', 1);
load_mw = table2array(load_mw);
load_mvar = table2array(load_mvar);
%states = generate_data(mpc, load_mw, load_mvar, 1, number + "_normal.csv");
%states = continuous_admittance(mpc, load_mw, load_mvar, 1, "14_continuous_admittance_evade.csv");
states = lines_out(mpc, load_mw, load_mvar, "14_lines_out_evade.csv");
%states = attacks("case14", load_mw, load_mvar, 1, "14_attacks.csv");
return
branch_r = table2array(readtable("14_r.csv", "NumHeaderLines", 1));
branch_x = table2array(readtable("14_x.csv", "NumHeaderLines", 1));
states = sample_scenarios(mpc, load_mw, load_mvar, branch_r, branch_x, 1, "14_cube_sampling.csv");


function states = attacks(casename, load_mw, load_mvar, start, filename)
    mpc = loadcase(casename);
    sim_mpc = loadcase(casename);
    results = runpf(mpc);
    states = get_header(results);
    define_constants;
    load_idx = 1;

    for l1=1:length(mpc.branch)
        for step=1:1000
            mpc.branch(:, BR_STATUS) = 1;
            mpc.branch(l1, BR_STATUS) = 0;
            mpc.bus(:, PD) = load_mw(load_idx, :);
            mpc.bus(:, QD) = load_mvar(load_idx, :);
            load_idx = load_idx + 1;
            results = runopf(mpc);
            if results.success == 1
                states = add_state(results, states);
            end
    
            % Make grid look like line is out
            for l2=1:length(mpc.branch)
                if l1 == l2 
                    continue
                end
    
                sim_mpc.branch(:, BR_STATUS) = 1;
                sim_mpc.branch(l1, BR_STATUS) = 0;
                sim_mpc.branch(l2, BR_STATUS) = 0;
                % Setup loads
                for bus=1:length(mpc.bus)
                    if bus == mpc.branch(l2, F_BUS) || bus == mpc.branch(l2, T_BUS)
                        % Inject false data
                        sim_mpc.bus(bus, PD) = mpc.bus(bus, PD);
                        sim_mpc.bus(bus, QD) = mpc.bus(bus, QD);
                    else
                        error = 0.95 + 0.1 * rand();
                        sim_mpc.bus(bus, PD) = mpc.bus(bus, PD) * error;
                        sim_mpc.bus(bus, QD) = mpc.bus(bus, QD) * error;
                    end
                end
                sim_results = runopf(sim_mpc);
                if sim_results.success == 0
                    continue
                end
    
                state = ["Date" "Time"];
                for i=1:length(mpc.bus)
                    if i == mpc.branch(l2, F_BUS) || i == mpc.branch(l2, T_BUS)
                        % Inject attacker values
                        grid = sim_results;
                    else
                        grid = results;
                    end
    
                    gen_idx = find(grid.gen(:, GEN_BUS) == i);
                    if ~isempty(gen_idx)
                        state(end + 1) = grid.gen(gen_idx, QG);
                        state(end + 1) = grid.gen(gen_idx, PG);
                    else
                        state(end + 1) = 0;
                        state(end + 1) = 0;
                    end
                    state(end + 1) = grid.bus(i, QD);
                    state(end + 1) = grid.bus(i, PD);
                    state(end + 1) = grid.bus(i, VA);
                    state(end + 1) = grid.bus(i, VM);
                end
                [ybus, yf, yt] = makeYbus(results);
                [sim_ybus, yf, yt] = makeYbus(sim_results);
                ybus(mpc.branch(l2, F_BUS), mpc.branch(l2, T_BUS)) = sim_ybus(mpc.branch(l2, F_BUS), mpc.branch(l2, T_BUS));
                ybus = reshape(full(ybus), 1, []);
                state = [state ybus];
                state(end + 1) = "yes";
                state(end + 1) = l2 - 1;
                states = [states;state];
    
                if mod(size(states, 1), 1000) == 0
                    writematrix(states, filename, 'WriteMode','append')
                    states = [];
                end
            end
        end
    end
    if length(states) > 0
        writematrix(states, filename, 'WriteMode','append')
    end
end

function states = sample_scenarios(mpc, load_mw, load_mvar, branch_r, branch_x, start, filename)
    results = runpf(mpc);
    states = get_header(results);
    define_constants;
    load_idx = 1;

    for i=start:670000
        mpc.bus(:, PD) = load_mw(load_idx, :);
        mpc.bus(:, QD) = load_mvar(load_idx, :);
        mpc.branch(:, BR_R) = branch_r(load_idx, :);
        mpc.branch(:, BR_X) = branch_x(load_idx, :);
        load_idx = load_idx + 1;
        results = runopf(mpc);
        if results.success == 1
            states = add_state(results, states);
            if mod(size(states, 1), 1000) == 0
                writematrix(states, filename, 'WriteMode','append')
                states = [];
            end
        end
    end

end
function states = lines_out(mpc, load_mw, load_mvar, filename)
    results = runpf(mpc);
    states = get_header(results);
    define_constants;
    load_idx = 250000;
    p_shift = 1 * (results.gen(:, PMAX) - results.gen(:, PMIN));
    q_shift = 1 * (results.gen(:, QMAX) - results.gen(:, QMIN));

    for line1=1:length(mpc.branch)
        for line2=1:length(mpc.branch)
            mpc.branch(:, BR_STATUS) = 1;
            mpc.branch(line1, BR_STATUS) = 0;
            mpc.branch(line2, BR_STATUS) = 0;
            for load=1:1000
                mpc.bus(:, PD) = load_mw(load_idx, :);
                mpc.bus(:, QD) = load_mvar(load_idx, :);
                load_idx = load_idx + 1;
                results = runopf(mpc);

                if results.success == 1
                    base_p = results.gen(:, PG);
                    base_q = results.gen(:, QG);
                    deltas = (rand([size(results.gen, 1), 1]) * 2 - 1) / size(results.gen, 1);
                    p_delta = p_shift .* deltas;
                    p_delta(1) = -(sum(p_delta) - p_delta(1));
                    q_delta = q_shift .* deltas;
                    q_delta(1) = -(sum(q_delta) - q_delta(1));
                    
                    mpc.gen(:, PG) = results.gen(:, PG) + p_delta;
                    mpc.gen(:, QG) = results.gen(:, QG) + q_delta;
                    pf = runpf(mpc);
                    if pf.success == 1
                        states = add_state_with_base(pf, states, base_p, base_q);
                        if mod(size(states, 1), 1000) == 0
                            writematrix(states, filename, 'WriteMode','append')
                            states = [];
                        end
                    end
                end
            end
        end
    end
end

function states = continuous_admittance(mpc, load_mw, load_mvar, start, filename)
    %  Ys = stat ./ (branch(:, BR_R) + 1j * branch(:, BR_X));  %% series admittance
    results = runpf(mpc);
    states = get_header(results);
    load_idx = start;
    define_constants;
    diverged = 0;
    p_shift = 1 * (results.gen(:, PMAX) - results.gen(:, PMIN));
    q_shift = 1 * (results.gen(:, QMAX) - results.gen(:, QMIN));

    base_r = mpc.branch(:, BR_R);
    base_x = mpc.branch(:, BR_X);
    for r=-0.1:0.01:1
        for x=-0.1:0.01:1
            for lin=1:length(mpc.branch)
                mpc.branch(:, BR_R) = base_r;
                mpc.branch(:, BR_X) = base_x;
                mpc.branch(lin, BR_R) = r;
                mpc.branch(lin, BR_X) = x;
                mpc.bus(:, PD) = load_mw(load_idx, :);
                mpc.bus(:, QD) = load_mvar(load_idx, :);
                load_idx = load_idx + 1;  
                results = runopf(mpc);

                if results.success == 1
                    base_p = results.gen(:, PG);
                    base_q = results.gen(:, QG);
                    deltas = (rand([size(results.gen, 1), 1]) * 2 - 1) / size(results.gen, 1);
                    p_delta = p_shift .* deltas;
                    p_delta(1) = -(sum(p_delta) - p_delta(1));
                    q_delta = q_shift .* deltas;
                    q_delta(1) = -(sum(q_delta) - q_delta(1));
                    
                    mpc.gen(:, PG) = results.gen(:, PG) + p_delta;
                    mpc.gen(:, QG) = results.gen(:, QG) + q_delta;
                    pf = runpf(mpc);
                    if pf.success == 1
                        states = add_state_with_base(pf, states, base_p, base_q);
                        if mod(size(states, 1), 1000) == 0
                            writematrix(states, filename, 'WriteMode','append')
                            states = [];
                        end
                    end
                end
            end
        end
    end
end

function states = generate_data(mpc, load_mw, load_mvar, start, filename)
    define_constants;
    results = runpf(mpc);    
    states = get_header(results);
    display(filename);
    for i=start:670000 %length(load_mw)
        mpc.bus(:, PD) = load_mw(i, :);
        mpc.bus(:, QD) = load_mvar(i, :);
        results = runopf(mpc);
        if results.success == 1

            states = add_state(results, states);
            if mod(size(states, 1), 1000) == 0
                writematrix(states, filename, 'WriteMode','append')
                states = [];
            end
        else
            diverged = diverged + 1;
        end
        
    end
end

function states = add_state(results, states)
    base_q = results.gen(:, QG);
    base_p = results.gen(:, PG);
   states = add_state_with_base(results, states, base_p, base_q);
end

function states = add_state_with_base(results, states, base_p, base_q)
    define_constants;
    state = ["Date" "Time"];
    for i=1:length(results.bus(:, VM))
        gen_idx = find(results.gen(:, GEN_BUS) == i);
        if ~isempty(gen_idx)
            state(end + 1) = results.gen(gen_idx, QG);
            state(end + 1) = base_q(gen_idx);
            state(end + 1) = results.gen(gen_idx, PG);
            state(end + 1) = base_p(gen_idx);
        else
            state(end + 1) = 0;
            state(end + 1) = 0;
            state(end + 1) = 0;
            state(end + 1) = 0;
        end
        state(end + 1) = results.bus(i, QD);
        state(end + 1) = results.bus(i, PD);
        state(end + 1) = results.bus(i, VA);
        state(end + 1) = results.bus(i, VM);
    end
    [ybus, yf, yt] = makeYbus(results);
    ybus = reshape(full(ybus), 1, []);
    state = [state ybus];
    state(end + 1) = "no";
    state(end + 1) = -1;
    states = [states;state];
end

function header = get_header(results)
    define_constants;
    header = ["" ""];
    for i = 1:length(results.bus(:, VM))
        header(end + 1) = i + " gen Mvar";
        header(end + 1) = i + " gen Mvar Base";
        header(end + 1) = i + " gen MW";
        header(end + 1) = i + " gen MW Base";
        header(end + 1) = i + " load Mvar";
        header(end + 1) = i + " load Mw";
        header(end + 1) = i + " Voltage angle";
        header(end + 1) = i + " Voltage mag";
    end
    for i=1:length(results.bus) * length(results.bus)
        header(end + 1) = "YBus " + i;
    end
    header(end + 1) = "Attack?";
    header(end + 1) = "Location";
end

function extract_loads(mpc, name)
    define_constants;
    mw =  mpc.bus(:, PD);
    mvar = mpc.bus(:, QD);
    loads = [mw mvar];
    loads = ["MW" "Mvar"; loads];
    writematrix(loads, name + "_base.csv")
end