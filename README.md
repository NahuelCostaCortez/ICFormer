# ICFormer

ICFormer is a novel Deep Learning model based on a Transformer encoder that leverages self-attention on the evolution of incremental capacity (IC) curves to accurately identify relevant changes in LIB degradation trajectories. It does not only detects knees, but also anticipates them while also outperforming state-of-the-art approaches in diagnosing degradation modes, making it a powerful tool for predicting battery health. ICFormer can provide valuable knowledge on the factors contributing to capacity loss and offer advanced insights for battery management and predictive maintenance strategies.

You can check our paper [here](https://www.sciencedirect.com/science/article/pii/S0378775323012867).

>[!WARNING]
>The code related to the model will no longer be mantained here but in the [rapidae](https://github.com/NahuelCostaCortez/rapidae) library.

# Files in this Repository
- \data: contains the synthetic test data and the real data corresponding to two commercial high-power graphite/LFP cells manufactured by A123 Systems: CReal#1 (ANR26650M1a, 2.3 Ah) and CReal#2 (ANR26650M1b, 2.5 Ah).
- \notebooks: contains notebooks explaining the training data and the experimental results reported in the paper.
- \saved: folder containing the trained models.
- \sweeps: contains the yaml config file for creating a sweep in wandb.
- ICFormer.py: proposed model architecture.
- train.py: script to train the model.
- utils.py: some helper functions.


The data used in this study is available for download:

[Diagnosis dataset](http://dx.doi.org/10.17632/bs2j56pn7y)

[Prognosis dataset](https://data.mendeley.com/datasets/6s6ph9n8zg/3)

Once you download the files:
- Open Matlab
- Execute the Read_me_VvsQmatrix.m. This will get the variables needed for the **diagnosis** dataset.
- Execute the Read_me_Dutymatrix.m. This will get the variables needed for the **prognosis** dataset. This is because the duty cycles are built over the paths generated for the diagnosis dataset.
- From these variables, you will need to save *Vi.mat*, *volt.mat*, *Q.mat* and *path_prognosis.mat*.
- To get *path_prognosis.mat*, you can execute the following snippet:
```
cycles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000];
% here, we will store the degradation modes associated with these cycles for each sample (duty cycle)
path_prognosis = zeros([4,length(cycles),length(key)]);

% iterate over duty_cycles
for duty_cycle = 1:length(key)
    duty_data_info = zeros([4,length(cycles)]);
    % iterate over cycles
    for index_cycle = 1:length(cycles)
        try
        % get the index of the diagnosis dataset that corresponds to this cycle
        voltage_index = Vi(duty_cycle,find(cyc==cycles(index_cycle)));
        % store the degradation modes
        duty_data_info(:,index_cycle) = pathinfo(voltage_index,:);
        end
    end
    path_prognosis(:,:,duty_cycle) = duty_data_info;
end
```
