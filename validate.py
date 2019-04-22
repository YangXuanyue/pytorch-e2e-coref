from runner import Runner

runner = Runner()
runner.load_ckpt()
# trainer.validate(name='valid', saves_results=True)
runner.evaluate()