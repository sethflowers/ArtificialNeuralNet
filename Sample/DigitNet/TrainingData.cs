//-----------------------------------------------------------------------
// <copyright file="TrainingData.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace DigitNet
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    /// <summary>
    /// Data used to train the neural nets.
    /// </summary>
    internal class TrainingData
    {
        /// <summary>
        /// Gets or sets the digit that is represented by the Data array.
        /// </summary>
        /// <value>
        /// The digit represented by the Data array.
        /// </value>
        public short Digit { get; set; }
        
        /// <summary>
        /// Gets or sets the image data that will be run through the neural nets.
        /// </summary>
        /// <value>
        /// The image data for a single digit.
        /// </value>
        public byte[] Data { get; set; }
    }
}
