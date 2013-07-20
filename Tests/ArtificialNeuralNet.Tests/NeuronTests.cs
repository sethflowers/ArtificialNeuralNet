//-----------------------------------------------------------------------
// <copyright file="NeuronTests.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet.Tests
{
    using System;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for the Neuron class.
    /// </summary>
    [TestClass]
    public class NeuronTests
    {
        /// <summary>
        /// Validates that the Inputs property is initialized by the constructor.
        /// </summary>
        [TestMethod]
        public void Constructor_InputsProperty_IsInitialized()
        {
            Assert.IsNotNull(new Neuron().Inputs);
        }

        /// <summary>
        /// Validates that the Outputs property is initialized by the constructor.
        /// </summary>
        [TestMethod]
        public void Constructor_OutputsProperty_IsInitialized()
        {
            Assert.IsNotNull(new Neuron().Outputs);
        }

        /// <summary>
        /// Validates that the Think method causes the neurons outputs to have the correct values set.
        /// This is a little difficult to test, because Think uses a sigmoid function, 
        /// which we pretty much need to copy in order to assert against.
        /// </summary>
        [TestMethod]
        public void Think_SetsValuesInOutputSynapses()
        {
            Neuron neuron = new Neuron();
       
            // Reset the bias, since we know the ctor provides a random bias.
            neuron.Bias = 0;

            // Add two inputs to the neuron.
            neuron.Inputs.Add(new Synapse { Value = 1, Weight = 4 });
            neuron.Inputs.Add(new Synapse { Value = 3, Weight = 2 });

            // Add two outputs to the neuron.
            // Causing the neuron to think should set the outputs values.
            neuron.Outputs.Add(new Synapse());
            neuron.Outputs.Add(new Synapse());

            // Execute the code to test.
            neuron.Think();

            //// Think sums the neurons bias (which we set to 0 to disregard)
            //// with each inputs weight times value.
            //// This means sum = bias + (1 * 4) + (2 * 3) = 10.
            //// It then runs the total through a sigmoid function.
            //// 1 / (1 + Math.Pow(Math.E, -sum))
            //// = 1 / (1 + e^-10)
            //// = 0.999954
            double expected = 1 / (1 + Math.Pow(Math.E, -10));

            // Validate that the outputs now have values.
            Assert.AreEqual(expected, neuron.Outputs[0].Value);
            Assert.AreEqual(expected, neuron.Outputs[1].Value);
        }
    }
}
