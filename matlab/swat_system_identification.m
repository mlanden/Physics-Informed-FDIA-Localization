swat = readtable("C:\Users\mland\OneDrive - Georgia Institute of Technology\Documents\Datasets\SWAT\SWaT_Dataset_Normal_v0.csv");
experiment = getExperiment(swat, 26, 2, [19 ], 30);
experiment.InputName = "Flow 301" ;% "flow 401"];
experiment.OutputName = "RO tank level";

size = 7200*2;
ze = experiment(1:size);
zv = experiment(size:2 * size);

delays=delayest(ze)
nn1=struc(1:5,1:5,delays(1));
%nn2=struc(1:5,1:5,delays(2));
selstruc(arxstruc(ze(:,:,1), zv(:,:,1), nn1))
%selstruc(arxstruc(ze(:,:,2),zv(:,:,2),nn2))
%marx = arx(ze, "na", 4, "nb", [2 1], "nk", [0 2]);
%mn4sid = n4sid(ze, 2:8, "InputDelay", [0 2]);
function experiment = getExperiment(data, index, target, inputs, outputs)
    start = 1;
    experiment = 0;
    i = 0;
    idx_data = table2array(data(:,index));
    while start < length(idx_data)
        i = i + 1;
        start = start - 1 + find(idx_data(start:end)==target, 1);
        if isempty(start)
            break
        end

        after = start + find(idx_data(start + 1:end)~=target, 1);
        
        if ~isempty(after)
            input_values = table2array(data(start:after - 1,inputs));
            output_values = table2array(data(start:after - 1,outputs));
        else
            input_values = table2array(data(start:end,inputs));
            output_values = table2array(data(start:end,outputs));
        end
        exp = iddata(output_values, input_values, 1);
        if ~isa(experiment, "iddata")
            experiment = exp;
        else
            experiment = [experiment; exp];
        end
        start = after + 1;
    end
    input_meean = mean(experiment.InputData)
    output_mean = mean(experiment.OutputData)
    experiment.InputData = experiment.InputData - ones(size(experiment.InputData, 1), 1) * mean(experiment.InputData);
    experiment.OutputData = experiment.OutputData - ones(size(experiment.OutputData, 1), 1) * mean(experiment.OutputData);
end 