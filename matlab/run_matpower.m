define_constants;
mpc = loadcase('case14');
%extract_loads(mpc, "14")
load_mw = readtable('14_MW.csv', 'NumHeaderLines', 1);
load_mvar = readtable('14_Mvar.csv', 'NumHeaderLines', 1);
states = generate_data(mpc, load_mw, load_mvar);

writematrix(states, "14_state.csv");

function extract_loads(mpc, name)
    define_constants;
    mw =  mpc.bus(:, PD);
    mvar = mpc.bus(:, QD);
    loads = [mw mvar];
    loads = ["MW" "Mvar"; loads];
    writematrix(loads, name + "_base.csv")
end

function states = generate_data(mpc, load_mw, load_mvar)
    define_constants;
    results = runpf(mpc);
    states = get_header(results);

    load_mw = table2array(load_mw);
    load_mvar = table2array(load_mvar);

    for i=1:length(load_mw)
        mpc.bus(:, PD) = load_mw(i, :);
        mpc.bus(:, QD) = load_mvar(i, :);
        results = runopf(mpc);
        states = add_state(results, states);
    end
end

function states = add_state(results, states)
    define_constants;
    state = ["Date" "Time"];
    for i=1:length(results.bus(:, VM))
        gen_idx = find(results.gen(:, GEN_BUS) == i);
        if ~isempty(gen_idx)
            state(end + 1) = results.gen(gen_idx, QG);
            state(end + 1) = results.gen(gen_idx, PG);
        else
            state(end + 1) = 0;
            state(end + 1) = 0;
        end
        state(end + 1) = results.bus(i, QD);
        state(end + 1) = results.bus(i, PD);
        state(end + 1) = results.bus(i, VA);
        state(end + 1) = results.bus(i, VM);
    end
    state(end + 1) = "no";
    states = [states;state];

end

function header = get_header(results)
    define_constants;
    header = ["" ""];
    for i = 1:length(results.bus(:, VM))
        header(end + 1) = i + " gen Mvar";
        header(end + 1) = i + " gen MW";
        header(end + 1) = i + " load Mvar";
        header(end + 1) = i + " load Mw";
        header(end + 1) = i + " Voltage angle";
        header(end + 1) = i + " Voltage mag";
    end
    header(end + 1) = "Attack?";
end