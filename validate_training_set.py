from runner import Runner

trainer = Runner()
trainer.load_ckpt()
trainer.evaluate(name='train')
