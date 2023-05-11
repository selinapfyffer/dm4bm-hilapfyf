# Implentation of a PID Controller to optimize the Efficiency during the Charging Process of a Storage Device filled with Phase Change Material
In order to master the energy transition in the future, traditional fossil fuels for heating buildings must be largely replaced. 
The latest research from ETH Zurich shows that half of Switzerland's electricity imports in winter could be covered by seasonal thermal energy storage. Phase change materials (PCM) are very well suited for a energy storage system. Compared to a conventional water storage tank, a storage tank filled with PCM has a three times higher energy density.

In this project, the charging behavior of a storage tank filled with phase change material is modeled in combination with a PID-Controller.

![zyklus](https://user-images.githubusercontent.com/90027713/204542465-0dc1ef17-8bbe-499c-9371-9ef3072c129b.PNG)
> Figure: Integration of the thermal energy storage in a building. The tank is filled in summer with lowcost surplus energy. In winter the energy is discharged in a costefficient way with a smart controller.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/selinapfyffer/dm4bm-pcm_melting/HEAD?labpath=https%3A%2F%2Fgithub.com%2Fselinapfyffer%2Fdm4bm-pcm_melting%2Fblob%2Fmain%2FImplementation_of_a_PID_Controller.ipynb)

## Abstract
This report examines the performance of a PID (Proportional-Integral-Derivative) controller and the melting process of a PCM (Phase Change Material) cell. The evaluation begins with an analysis of the PID controller's response in various scenarios, revealing suboptimal parameter settings leading to a persistent offset between the controlled signal and the setpoint. The K_p parameter contributes to this offset, while the K_i parameter results in frequent valve switching.

Next, the charging time of the PCM cell is investigated, showing a total duration of approximately 25 hours. Notably, the temperature behavior within the cell demonstrates a significant discrepancy between the charging times of the top and bottom layers. The first layer achieves full charge in less than 5 hours, while the bottom layer requires over 20 hours. This discrepancy is attributed to the omission of convection in the heat transfer calculation within the PCM cell, highlighting the need for a more accurate model that considers both conduction and convection.

In conclusion, the results indicate that the PID controller effectively maintains a relatively constant inlet temperature during the PCM cell charging process. However, the suboptimal performance of the controller and the absence of convection modeling suggest the need for further improvements. This includes optimizing the controller parameters and implementing a more comprehensive model to account for convection, ultimately enhancing the efficiency of the system.

## Written report
The report will be written in *Jupyter* notebook, posted on *GitHub.com* and liked to *MyBinder.org*.

## Structure of the report:
[1.1] Introduction

[1.2] Problem Definition

[1.3] Physical Model

[1.4] PID-Controller Theory

[1.5] Initial System Behavior

[1.6] Parameter Study

[1.7] Conclusion

[1.8] Outlook

**Licence**

Code is released under [MIT Lincence](https://choosealicense.com/licenses/mit/).

[![Creative Commons License](http://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
