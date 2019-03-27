from runner import Runner

trainer = Runner()
trainer.load_ckpt()
# trainer.validate(name='valid', saves_results=True)
trainer.test()