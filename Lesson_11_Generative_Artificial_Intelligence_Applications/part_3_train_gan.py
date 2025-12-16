def train_gan(generator, discriminator, data_loader, epochs, noise_size):
    criterion = nn.BCELoss()
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_data, _ in data_loader:
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            optimizer_disc.zero_grad()
            outputs = discriminator(real_data)
            disc_loss_real = criterion(outputs, real_labels)
            disc_loss_real.backward()

            noise = torch.randn(batch_size, noise_size)
            fake_data = generator(noise)
            outputs = discriminator(fake_data.detach())
            disc_loss_fake = criterion(outputs, fake_labels)
            disc_loss_fake.backward()
            optimizer_disc.step()

            # Train Generator
            optimizer_gen.zero_grad()
            outputs = discriminator(fake_data)
            gen_loss = criterion(outputs, real_labels)
            gen_loss.backward()
            optimizer_gen.step()
