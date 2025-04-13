const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function runSwap() {
  const form = new FormData();
  form.append('source', fs.createReadStream('./test/ref.png'));  // Face to use
  form.append('target', fs.createReadStream('./test/main.png')); // Where to place the face
  

  
  try {
    const response = await axios.post(
      'https://v1yfznzb7lkce6-7860.proxy.runpod.net/swap',
      form,
      {
        headers: form.getHeaders(),
        responseType: 'arraybuffer'
      }
    );
    
    fs.writeFileSync('./output/result.png', response.data);
    console.log('Face swap complete. Saved as result.png');
  } catch (error) {
    console.error('Error during face swap:', error);
  }
}

runSwap();