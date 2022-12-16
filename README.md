# MMEngine Template

**MMEngine Template** is a template for the best practices in MMEngine. Config&Resgitry in [MMEngine](https://github.com/open-mmlab/mmengine) provides an elegant solution for developers to customize their modules and record experimental information. However, due to the learning curve of the Config&Registry system, developers often encounter some problems during developping. Therefore, we provide the **MMEngine Template** to guide developers in achieving the best practices in various deep learning fields based on MMEngine.

You may wonder why there is a need for **MMEngine Template** since the downstream repositories (MMDet, MMCls) of OpenMMLab have already provided a "template". Actually, **MMEngine Template** provides a more lightweight development standard compared to the downstream algorithm repositories. **MMEngine Template** is designed to provide a generic development template to simplify the development process based on MMEngine.

Compared with downstream repositories, **MMEngine Template**:

- A more concise directory structure for easier maintenance. Since most of developers do not need to customize lots of datasets, hooks, loops, etc., there is not need to create too much nested directory.
- Looser standards for the data flow. OpenMMLab series repositories need to obey a more strict dataflow based on `mmengine.structures.BaseDataElement`, which is not necessary for most individual developers. Therefore, **MMEngine Template** does not require developers to format the data to `mmengine.structures.BaseDataElement` instance in data flow.
- Looser code standards, developers no longer struggle to fix mypy errors.

Since developers often meet the error of "Unregistered module xxx" for the lack of triggering the registerring, **MMEngine Template** also will register the module automatically if developers follow the default [directory structure](#directory-structure).

## Installation

1. Follow the [official guide](https://pytorch.org/get-started/locally/) to install PyTorch.
2. Install MMEngine
   ```
   pip install -U openmim
   mim install mmengine
   python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
   ```

## Directory structure

```bash
├── configs                                 Commonly used base config file.
├── demo
│   ├── mmengine_template_demo.py           General demo script
├── mmengine_template
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── datasets.py                     Customize your dataset here
│   │   └── transforms.py                   Customize your data transform here
│   ├── engine
│   │   ├── __init__.py
│   │   ├── hooks.py                        Customize your hooks here
│   │   ├── optimizers.py                   Less commonly used. Customize your optimizer here
│   │   ├── optim_wrappers.py               Less commonly used. Customize your optimizer wrapper here
│   │   ├── optim_wrapper_constructors.py   Less commonly used. Customize your optimizer wrapper constructor here
│   │   └── schedulers.py                   Customize your lr/momentum scheduler here
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py                    Less commonly used. Customize your evaluator here
│   │   └── metrics.py                      Customize your metric here.
│   ├── infer
│   │   ├── inference.py                    Used for demo script. Customize your inferencer here
│   │   └── __init__.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── model.py                        Customize your model here.
│   │   ├── weight_init.py                  Less commonly used here. Customize your initializer here.
│   │   └── wrappers.py                     Less commonly used here. Customize your wrapper here.
│   ├── registry.py
│   └── version.py
├── tools                                   General train/test script
```

## How to use

Assuming you have already understood the basic process of developing based on MMEngine, then when developing based on **MMEngine Template**, you need to customize your module in `datasets/datasets.py`, `datasets/transform.py`, `models/models.py`, etc., and training pipelines will be automatically built in `mmengine.Runner`. **MMEngine Template** provides a general training/testing/inferring script in `tools` and `demo`, and you can directly use them in command line.

For advanced users, you may need to customize more components and register more modules. When developing, remember to update the locations parameter in registry.py when adding new modules to ensure that the newly added modules are correctly registered.

**MMEngine Template** will continuously update new branches to support various deep learning tasks in different fields, so stay tuned.
